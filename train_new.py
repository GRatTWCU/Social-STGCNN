import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx

# utilsモジュールとmodelモジュールは外部からインポートされる想定
# from utils import *
# from model import *
# metricsモジュールからのインポートは、必要な関数がこのファイル内に統合されたため削除します。
import pickle
import argparse
from torch import autograd
import torch.optim.lr_scheduler as lr_scheduler

# ADE (Average Displacement Error) の計算
def ade(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        # predAll[s]は (pred_seq_len, num_peds, 2) の形状
        # targetAll[s]も (pred_seq_len, num_peds, 2) の形状
        # np.swapaxes(..., 0, 1) で (num_peds, pred_seq_len, 2) に変換
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)

        N = pred.shape[0] # 歩行者数
        T = pred.shape[1] # シーケンス長
        sum_ = 0
        for i in range(N): # 各歩行者について
            for t in range(T): # 各タイムステップについて
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N*T) # 各サンプルでの平均ADE

    return sum_all/All # 全サンプルでの平均ADE

# FDE (Final Displacement Error) の計算
def fde(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        # predAll[s]は (pred_seq_len, num_peds, 2) の形状
        # targetAll[s]も (pred_seq_len, num_peds, 2) の形状
        # np.swapaxes(..., 0, 1) で (num_peds, pred_seq_len, 2) に変換
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        N = pred.shape[0] # 歩行者数
        T = pred.shape[1] # シーケンス長
        sum_ = 0
        for i in range(N): # 各歩行者について
            for t in range(T-1,T): # 最終タイムステップについてのみ
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N) # 各サンプルでの平均FDE

    return sum_all/All # 全サンプルでの平均FDE

# シーケンスデータをノード形式に変換（この関数は直接は使用されないかもしれませんが、元のmetrics.pyに存在しました）
def seq_to_nodes(seq_):
    max_nodes = seq_.shape[1] # グラフ内の歩行者数
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]

    V = np.zeros((seq_len,max_nodes,2))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        for h in range(len(step_)):
            V[s,h,:] = step_[h]

    return V.squeeze()

# 相対座標を絶対座標に変換
def nodes_rel_to_nodes_abs(nodes,init_node):
    # nodes: (seq_len, num_nodes, 2) 相対座標
    # init_node: (num_nodes, 2) 各ノードの初期絶対座標 (観測シーケンスの最後の位置)
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]): # シーケンス長
        for ped in range(nodes.shape[1]): # 歩行者数
            # 累積和 + 初期位置
            nodes_[s,ped,:] = np.sum(nodes[:s+1,ped,:],axis=0) + init_node[ped,:]

    return nodes_.squeeze()

# ゼロに近いかどうかを判定（この関数は直接は使用されないかもしれませんが、元のmetrics.pyに存在しました）
def closer_to_zero(current,new_v):
    dec =  min([(abs(current),current),(abs(new_v),new_v)])[1]
    if dec != current:
        return True
    else:
        return False

# Bivariate Gaussian NLL損失の計算
def bivariate_loss(V_pred, V_trgt):
    # print文はデバッグ用なので、本番コードでは削除またはコメントアウトしても良い
    # print(f"bivariate_loss input V_pred.shape={V_pred.shape}, V_trgt.shape={V_trgt.shape}")

    # train/vald関数で既に適切なリシェイプが行われているため、
    # ここではV_predとV_trgtが (N, feature_dim) の形状で来ると仮定します。
    # したがって、以前のpermuteロジックは不要です。

    # V_pred と V_trgt が (N, feature_dim) の形状で来ると仮定
    # そのため、[:, :, 0] のようなインデックスは不要になり、[:, 0] となります。
    normx = V_trgt[:, 0] - V_pred[:, 0]
    normy = V_trgt[:, 1] - V_pred[:, 1]

    sx = torch.exp(V_pred[:, 2])
    sy = torch.exp(V_pred[:, 3])
    corr = torch.tanh(V_pred[:, 4])

    sxsy = sx * sy
    z = (normx / sx)**2 + (normy / sy)**2 - 2 * (corr * normx * normy / sxsy)
    negRho = 1 - corr**2

    result = torch.exp(-z / (2 * negRho))
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))
    epsilon = 1e-20 # ゼロ除算やlog(0)を防ぐための小さな値

    result = result / denom
    result = -torch.log(torch.clamp(result, min=epsilon)) # log(0)を避けるためにclamp
    return torch.mean(result)

# graph_loss関数: bivariate_lossを呼び出すラッパー
def graph_loss(V_pred, V_trgt):
    # V_predとV_trgtは既に (N, feature_dim) の形状にリシェイプされていると仮定
    return bivariate_loss(V_pred, V_trgt)

# ADEとFDEを計算するメイン関数
def calculate_ade_fde(V_pred_original_shape, V_tr_original_shape, obs_traj, pred_traj_gt):
    # V_pred_original_shape: (batch_size, pred_seq_len, num_pedestrians, 5)
    # V_tr_original_shape: (batch_size, pred_seq_len, num_pedestrians, 2) (これはpred_traj_gtと同じ意味合い)
    # obs_traj: (batch_size, obs_seq_len, num_pedestrians, 2)
    # pred_traj_gt: (batch_size, pred_seq_len, num_pedestrians, 2)

    # 予測された平均座標 (mu_x, mu_y) を抽出
    V_pred_means = V_pred_original_shape[..., :2] # shape: (B, T_pred, N, 2)

    # 観測シーケンスの最後の絶対位置を取得
    last_obs_pos = obs_traj[:, -1, :, :] # shape: (B, N, 2)

    all_pred_abs = []
    all_target_abs = []
    counts = []

    # バッチ内の各サンプルをループ
    for i in range(V_pred_means.shape[0]): # batch_size
        batch_V_pred_means = V_pred_means[i] # (T_pred, N, 2)
        batch_pred_traj_gt = pred_traj_gt[i] # (T_pred, N, 2)
        batch_last_obs_pos = last_obs_pos[i] # (N, 2)

        num_peds_in_batch = batch_V_pred_means.shape[1] # このサンプルでの歩行者数

        # 相対予測を絶対予測に変換するためにnumpyに変換
        batch_V_pred_means_np = batch_V_pred_means.cpu().numpy()
        batch_last_obs_pos_np = batch_last_obs_pos.cpu().numpy()
        batch_pred_traj_gt_np = batch_pred_traj_gt.cpu().numpy()

        # nodes_rel_to_nodes_abs を呼び出して絶対座標を取得
        # nodes_rel_to_nodes_abs は (seq_len, num_nodes, 2) を期待する
        pred_abs_np = nodes_rel_to_nodes_abs(batch_V_pred_means_np, batch_last_obs_pos_np)

        # ADE/FDE関数に渡すために結果をリストに追加
        all_pred_abs.append(pred_abs_np)
        all_target_abs.append(batch_pred_traj_gt_np)
        counts.append(num_peds_in_batch)

    # 全体のADEとFDEを計算
    ade_val = ade(all_pred_abs, all_target_abs, counts)
    fde_val = fde(all_pred_abs, all_target_abs, counts)

    return ade_val, fde_val


def train(epoch, model, loader_train, optimizer, args, device):
    model.train()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    turn_point = int(loader_len/args.batch_size)*args.batch_size + loader_len%args.batch_size - 1

    total_ade = 0
    total_fde = 0
    metric_batch_count = 0

    for cnt, batch in enumerate(loader_train):
        batch_count += 1

        # データを取得
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask, V_obs, A_obs, V_tr, A_tr = batch

        optimizer.zero_grad()

        # フォワードパス
        # V_obs = (batch, seq, node, feat)
        # V_obs_tmp = (batch, feat, seq, node)
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())

        # V_predの形状を (batch_size, 5, pred_seq_len, num_pedestrians) から
        # (batch_size, pred_seq_len, num_pedestrians, 5) に変換
        V_pred = V_pred.permute(0, 2, 3, 1)

        # ここでV_predとV_trをbivariate_lossが期待する形状にリシェイプ
        # V_pred.shape: (batch_size, pred_seq_len, num_pedestrians, 5)
        # V_tr.shape: (batch_size, pred_seq_len, num_pedestrians, 2)

        # graph_lossに渡す前にフラット化
        # -1は自動的に計算される次元のサイズ
        V_pred_reshaped = V_pred.contiguous().view(-1, V_pred.shape[-1])
        V_tr_reshaped = V_tr.contiguous().view(-1, V_tr.shape[-1])

        # リシェイプされたテンソルをgraph_lossに渡す
        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = graph_loss(V_pred_reshaped, V_tr_reshaped)
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss += l
        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            loss.backward()

            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()

            # メトリクスを計算 (calculate_ade_fdeにはリシェイプ前の元の4次元テンソルを渡す)
            with torch.no_grad():
                ade, fde = calculate_ade_fde(V_pred, V_tr, obs_traj, pred_traj_gt)
                total_ade += ade
                total_fde += fde
                metric_batch_count += 1

            loss_batch += loss.item()
            print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch/batch_count,
                  '\t ADE:', total_ade/metric_batch_count, '\t FDE:', total_fde/metric_batch_count)

    avg_ade = total_ade / metric_batch_count if metric_batch_count > 0 else 0
    avg_fde = total_fde / metric_batch_count if metric_batch_count > 0 else 0

    return loss_batch/batch_count, avg_ade, avg_fde

def vald(epoch, model, loader_val, args, device):
    model.eval()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_val)
    turn_point = int(loader_len/args.batch_size)*args.batch_size + loader_len%args.batch_size - 1

    total_ade = 0
    total_fde = 0
    metric_batch_count = 0

    with torch.no_grad():
        for cnt, batch in enumerate(loader_val):
            batch_count += 1

            # データを取得
            batch = [tensor.to(device) for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
             loss_mask, V_obs, A_obs, V_tr, A_tr = batch

            V_obs_tmp = V_obs.permute(0, 3, 1, 2)

            V_pred, _ = model(V_obs_tmp, A_obs.squeeze())

            V_pred = V_pred.permute(0, 2, 3, 1)

            # ここでV_predとV_trをbivariate_lossが期待する形状にリシェイプ
            # V_pred.shape: (batch_size, pred_seq_len, num_pedestrians, 5)
            # V_tr.shape: (batch_size, pred_seq_len, num_pedestrians, 2)

            # graph_lossに渡す前にフラット化
            V_pred_reshaped = V_pred.contiguous().view(-1, V_pred.shape[-1])
            V_tr_reshaped = V_tr.contiguous().view(-1, V_tr.shape[-1])

            if batch_count % args.batch_size != 0 and cnt != turn_point:
                l = graph_loss(V_pred_reshaped, V_tr_reshaped)
                if is_fst_loss:
                    loss = l
                    is_fst_loss = False
                else:
                    loss += l
            else:
                loss = loss / args.batch_size
                is_fst_loss = True

                # メトリクスを計算 (calculate_ade_fdeにはリシェイプ前の元の4次元テンソルを渡す)
                ade, fde = calculate_ade_fde(V_pred, V_tr, obs_traj, pred_traj_gt)
                total_ade += ade
                total_fde += fde
                metric_batch_count += 1

                loss_batch += loss.item()
                print('VALD:', '\t Epoch:', epoch, '\t Loss:', loss_batch/batch_count,
                      '\t ADE:', total_ade/metric_batch_count, '\t FDE:', total_fde/metric_batch_count)

    avg_ade = total_ade / metric_batch_count if metric_batch_count > 0 else 0
    avg_fde = total_fde / metric_batch_count if metric_batch_count > 0 else 0

    return loss_batch/batch_count, avg_ade, avg_fde


def main():
    parser = argparse.ArgumentParser()

    # Model specific parameters
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    parser.add_argument('--n_stgcnn', type=int, default=1, help='Number of ST-GCNN layers')
    parser.add_argument('--n_txpcnn', type=int, default=5, help='Number of TXPCNN layers')
    parser.add_argument('--kernel_size', type=int, default=3)

    # Data specific parameters
    parser.add_argument('--obs_seq_len', type=int, default=8)
    parser.add_argument('--pred_seq_len', type=int, default=12)
    parser.add_argument('--dataset', default='eth',
                        help='eth,hotel,univ,zara1,zara2,nuscenes_mini')

    # Training specific parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=250,
                        help='number of epochs')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='gradient clipping')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_sh_rate', type=int, default=150,
                        help='number of steps to drop the lr')
    parser.add_argument('--use_lrschd', action="store_true", default=False,
                        help='Use lr rate scheduler')
    parser.add_argument('--tag', default='tag',
                        help='personal tag for the model')

    # nuScenes specific parameters
    parser.add_argument('--nuscenes_setup', action="store_true", default=False,
                        help='Run nuScenes setup before training')
    parser.add_argument('--nuscenes_mode', choices=['raw', 'processed', 'dummy'], default='dummy',
                        help='nuScenes data conversion mode')
    parser.add_argument('--nuscenes_input_path', type=str, default='./nuscenes_mini/',
                        help='Input path for nuScenes data')

    args = parser.parse_args()

    print('*'*30)
    print("Training initiating....")
    print(args)

    # nuScenes setup if requested
    if args.nuscenes_setup or args.dataset == 'nuscenes_mini':
        print("Setting up nuScenes_mini data...")
        try:
            # complete_nuscenes_setup.pyは外部ファイルなので、適切にインポートされていることを確認
            from complete_nuscenes_setup import setup_directories, convert_nuscenes_raw_to_eth, convert_processed_data_to_eth, create_dummy_data, verify_data_format

            # Setup directories
            setup_directories()

            # Convert data based on mode
            success = False
            output_path = './datasets/nuscenes_mini/'

            if args.nuscenes_mode == 'raw':
                success = convert_nuscenes_raw_to_eth(args.nuscenes_input_path, output_path)
            elif args.nuscenes_mode == 'processed':
                success = convert_processed_data_to_eth(args.nuscenes_input_path, output_path)
            elif args.nuscenes_mode == 'dummy':
                create_dummy_data()
                success = True

            if success:
                verify_data_format(output_path)
                print("nuScenes_mini setup completed successfully!")
            else:
                print("nuScenes_mini setup failed. Exiting...")
                return

        except ImportError as e:
            print(f"Error importing nuScenes setup module: {e}")
            print("Please make sure complete_nuscenes_setup.py is in the same directory.")
            return
        except Exception as e:
            print(f"Error during nuScenes setup: {e}")
            return

    # Data prep
    obs_seq_len = args.obs_seq_len
    pred_seq_len = args.pred_seq_len
    data_set = './datasets/' + args.dataset + '/'

    # Check if dataset exists
    if not os.path.exists(data_set):
        print(f"Dataset path {data_set} does not exist!")
        if args.dataset == 'nuscenes_mini':
            print("Please run with --nuscenes_setup flag to setup the dataset first.")
            print("Example: python train.py --dataset nuscenes_mini --nuscenes_setup --nuscenes_mode dummy")
        return

    # Check train and val directories
    train_path = data_set + 'train/'
    val_path = data_set + 'val/'

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"Train ({train_path}) or Val ({val_path}) directory missing!")
        if args.dataset == 'nuscenes_mini':
            print("Please run with --nuscenes_setup flag to setup the dataset first.")
        return

    # Check if there are data files
    train_files = [f for f in os.listdir(train_path) if f.endswith('.txt')]
    val_files = [f for f in os.listdir(val_path) if f.endswith('.txt')]

    if not train_files or not val_files:
        print(f"No data files found in train ({len(train_files)}) or val ({len(val_files)}) directories!")
        if args.dataset == 'nuscenes_mini':
            print("Please run with --nuscenes_setup flag to setup the dataset first.")
        return

    print(f"Found {len(train_files)} training files and {len(val_files)} validation files")

    try:
        # TrajectoryDatasetは外部ファイル (utils.py) からインポートされる想定
        # social_stgcnnは外部ファイル (model.py) からインポートされる想定
        # これらのインポートが正しく行われていることを確認してください。
        from utils import TrajectoryDataset
        from model import social_stgcnn

        dset_train = TrajectoryDataset(
                train_path,
                obs_len=obs_seq_len,
                pred_len=pred_seq_len,
                skip=1, norm_lap_matr=True)

        loader_train = DataLoader(
                dset_train,
                batch_size=1,  # This is irrelevant to the args batch size parameter
                shuffle=True,
                num_workers=0)

        dset_val = TrajectoryDataset(
                val_path,
                obs_len=obs_seq_len,
                pred_len=pred_seq_len,
                skip=1, norm_lap_matr=True)

        loader_val = DataLoader(
                dset_val,
                batch_size=1,  # This is irrelevant to the args batch size parameter
                shuffle=False,
                num_workers=0)  # Changed from 1 to 0 to avoid multiprocessing issues

        print(f"Train dataset: {len(dset_train)} samples")
        print(f"Val dataset: {len(dset_val)} samples")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check your data format and paths.")
        return

    # Defining the model
    model = social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                         output_feat=args.output_size, seq_len=args.obs_seq_len,
                         kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len).to(device)

    # Training settings
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    if args.use_lrschd:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)

    # Update checkpoint directory to include dataset name
    checkpoint_dir = f'./checkpoint/{args.dataset}_{args.tag}/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    print('Data and model loaded')
    print('Checkpoint dir:', checkpoint_dir)

    # Training
    metrics = {'train_loss': [], 'val_loss': [], 'train_ade': [], 'val_ade': [],
               'train_fde': [], 'val_fde': []}
    constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999,
                       'min_ade': 9999999999999999, 'min_fde': 9999999999999999}

    print('Training started ...')
    for epoch in range(args.num_epochs):
        train_loss, train_ade, train_fde = train(epoch, model, loader_train, optimizer, args, device)
        val_loss, val_ade, val_fde = vald(epoch, model, loader_val, args, device)

        if args.use_lrschd:
            scheduler.step()

        # Save metrics
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_ade'].append(train_ade)
        metrics['val_ade'].append(val_ade)
        metrics['train_fde'].append(train_fde)
        metrics['val_fde'].append(val_fde)

        # Update best metrics
        if val_loss < constant_metrics['min_val_loss']:
            constant_metrics['min_val_loss'] = val_loss
            constant_metrics['min_val_epoch'] = epoch

        if val_ade < constant_metrics['min_ade']:
            constant_metrics['min_ade'] = val_ade

        if val_fde < constant_metrics['min_fde']:
            constant_metrics['min_fde'] = val_fde

        # Save model checkpoint
        if epoch % 20 == 0:  # Save every 20 epochs
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_ade': val_ade,
                'val_fde': val_fde,
                'metrics': metrics,
                'constant_metrics': constant_metrics
            }, checkpoint_dir + f'val_best_{epoch}.pth')

        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Train ADE: {train_ade:.4f}, Val ADE: {val_ade:.4f}')
        print(f'Train FDE: {train_fde:.4f}, Val FDE: {val_fde:.4f}')
        print(f'Best Val Loss: {constant_metrics["min_val_loss"]:.4f} at epoch {constant_metrics["min_val_epoch"]}')
        print(f'Best ADE: {constant_metrics["min_ade"]:.4f}, Best FDE: {constant_metrics["min_fde"]:.4f}')
        print('-' * 50)

    # Save final results
    print('Training completed!')
    print(f'Final Results:')
    print(f'Best Validation Loss: {constant_metrics["min_val_loss"]:.4f} at epoch {constant_metrics["min_val_epoch"]}')
    print(f'Best ADE: {constant_metrics["min_ade"]:.4f}')
    print(f'Best FDE: {constant_metrics["min_fde"]:.4f}')

    # Save final model and metrics
    torch.save({
        'epoch': args.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'constant_metrics': constant_metrics
    }, checkpoint_dir + 'final_model.pth')

    with open(checkpoint_dir + 'metrics.pkl', 'wb') as fp:
        pickle.dump(metrics, fp)

    with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as fp:
        pickle.dump(constant_metrics, fp)

if __name__ == '__main__':
    main()