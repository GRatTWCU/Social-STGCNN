import os
import math
import sys
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist
from utils import *
from metrics import *
from model import social_stgcnn
import copy
import matplotlib.pyplot as plt # ← 追加：可視化ライブラリ

# ← 追加：ここから可視化関数の定義
def visualize_scene(obs_traj, pred_traj_gt, pred_traj_best, save_path):
    """
    軌道を可視化して画像として保存する関数
    Args:
    - obs_traj (np.array): 過去の軌道 (obs_len, num_peds, 2)
    - pred_traj_gt (np.array): 正解の未来軌道 (pred_len, num_peds, 2)
    - pred_traj_best (np.array): 最も良かった予測未来軌道 (pred_len, num_peds, 2)
    - save_path (str): 保存先のファイルパス
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    num_peds = obs_traj.shape[1]

    for i in range(num_peds):
        # 過去の軌道を描画 (青の破線)
        ax.plot(obs_traj[:, i, 0], obs_traj[:, i, 1], 'b--')
        # 軌道の開始点
        ax.plot(obs_traj[0, i, 0], obs_traj[0, i, 1], 'bo')

        # 正解の未来軌道を描画 (緑の実線)
        # 過去の最後の点から繋げる
        full_gt_traj = np.concatenate([obs_traj[-1:, i, :], pred_traj_gt[:, i, :]])
        ax.plot(full_gt_traj[:, 0], full_gt_traj[:, 1], 'g-', linewidth=2, label='Ground Truth' if i == 0 else "")

        # 予測の未来軌道を描画 (赤の点線)
        # 過去の最後の点から繋げる
        full_pred_traj = np.concatenate([obs_traj[-1:, i, :], pred_traj_best[:, i, :]])
        ax.plot(full_pred_traj[:, 0], full_pred_traj[:, 1], 'r:', linewidth=2, label='Prediction' if i == 0 else "")

    ax.set_title(os.path.basename(save_path))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

    plt.savefig(save_path, dpi=300)
    plt.close(fig)
# ← 追加：ここまで可視化関数の定義

def test(args, KSTEPS=20): # ← 変更：argsを受け取る
    global loader_test, model
    model.eval()
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step = 0
    for batch in loader_test:
        step += 1
        #Get data
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask,V_obs,A_obs,V_tr,A_tr = batch

        num_of_objs = obs_traj_rel.shape[1]

        #Forward
        V_obs_tmp =V_obs.permute(0,3,1,2)
        V_pred,_ = model(V_obs_tmp,A_obs.squeeze())
        V_pred = V_pred.permute(0,2,3,1)

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred,V_tr =  V_pred[:,:num_of_objs,:],V_tr[:,:num_of_objs,:]
        
        sx = torch.exp(V_pred[:,:,2]) #sx
        sy = torch.exp(V_pred[:,:,3]) #sy
        corr = torch.tanh(V_pred[:,:,4]) #corr
        
        cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],2,2).to(device)
        cov[:,:,0,0]= sx*sx
        cov[:,:,0,1]= corr*sx*sy
        cov[:,:,1,0]= corr*sx*sy
        cov[:,:,1,1]= sy*sy
        mean = V_pred[:,:,0:2]
        
        mvnormal = torchdist.MultivariateNormal(mean,cov)

        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                             V_x[0,:,:].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                             V_x[-1,:,:].copy())
        
        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        ade_ls = {}
        fde_ls = {}
        for n in range(num_of_objs):
            ade_ls[n]=[]
            fde_ls[n]=[]

        for k in range(KSTEPS):
            V_pred_sample = mvnormal.sample()
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred_sample.data.cpu().numpy().squeeze().copy(),
                                                     V_x[-1,:,:].copy())
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))
            
            for n in range(num_of_objs):
                pred = [V_pred_rel_to_abs[:,n:n+1,:]]
                target = [V_y_rel_to_abs[:,n:n+1,:]]
                number_of = [1]
                ade_ls[n].append(ade(pred,target,number_of))
                fde_ls[n].append(fde(pred,target,number_of))
        
        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

        # ← 追加：ここから可視化処理
        if args.visualize:
            # 20個の予測サンプルの中から、シーン全体で最もADEが良かったものを探す
            all_samples_ade = []
            for k in range(KSTEPS):
                pred_k = raw_data_dict[step]['pred'][k]
                target_gt = raw_data_dict[step]['trgt']
                # このサンプルkのシーン全体のADEを計算
                scene_ade = np.mean(np.linalg.norm(pred_k - target_gt, axis=2))
                all_samples_ade.append(scene_ade)
            
            # 最もADEが小さかったサンプルのインデックスを取得
            best_sample_idx = np.argmin(all_samples_ade)
            best_pred_traj = raw_data_dict[step]['pred'][best_sample_idx]
            
            # 可視化関数を呼び出す
            obs_traj_to_plot = raw_data_dict[step]['obs']
            gt_traj_to_plot = raw_data_dict[step]['trgt']
            
            # 保存先フォルダの準備
            save_dir = os.path.join(os.path.dirname(args.model_path), "visualizations")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_file_path = os.path.join(save_dir, f"scene_{step:04d}.png")
            visualize_scene(obs_traj_to_plot, gt_traj_to_plot, best_pred_traj, save_file_path)
            
            # 可視化しすぎないように、一定数で終了する
            if step > 50:
                print("Generated 50 visualization images. Exiting test loop.")
                break
        # ← 追加：ここまで可視化処理

    ade_ = sum(ade_bigls)/len(ade_bigls)
    fde_ = sum(fde_bigls)/len(fde_bigls)
    return ade_,fde_,raw_data_dict


def main(args): # ← 変更：argsを受け取る
    paths = ['./checkpoint/*social-stgcnn*']
    KSTEPS=20

    print("*"*50)
    print('Number of samples:',KSTEPS)
    print("*"*50)

    for feta in range(len(paths)):
        ade_ls = [] 
        fde_ls = [] 
        path = paths[feta]
        exps = glob.glob(path)
        print('Model being tested are:',exps)

        for exp_path in exps:
            print("*"*50)
            print("Evaluating model:",exp_path)

            model_path = exp_path+'/val_best.pth'
            args_path = exp_path+'/args.pkl'

            # ← 追加：コマンドライン引数でモデルパスが指定されたら、そちらを優先
            if args.model_path != "":
                model_path = args.model_path

            # 可視化のためにargsにモデルパスを追加
            args.model_path = model_path

            with open(args_path,'rb') as f: 
                args_saved = pickle.load(f)

            stats= exp_path+'/constant_metrics.pkl'
            with open(stats,'rb') as f: 
                cm = pickle.load(f)
            print("Stats:",cm)

            #Data prep      
            obs_seq_len = args_saved.obs_seq_len
            pred_seq_len = args_saved.pred_seq_len
            data_set = './datasets/'+args_saved.dataset+'/'

            dset_test = TrajectoryDataset(
                    data_set+'test/',
                    obs_len=obs_seq_len,
                    pred_len=pred_seq_len,
                    skip=1,norm_lap_matr=True)

            loader_test = DataLoader(
                    dset_test,
                    batch_size=1, # This is irrelative to the args batch size parameter
                    shuffle =False,
                    num_workers=1)

            #Defining the model 
            model = social_stgcnn(n_stgcnn =args_saved.n_stgcnn,n_txpcnn=args_saved.n_txpcnn,
            output_feat=args_saved.output_size,seq_len=args_saved.obs_seq_len,
            kernel_size=args_saved.kernel_size,pred_seq_len=args_saved.pred_seq_len).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))

            ade_ =999999
            fde_ =99999fde_ =999999
            print("Testing ....")
            ad,fd,raw_data_dic_= test(args, KSTEPS=KSTEPS) # ← 変更: argsを渡す
            ade_= min(ade_,ad)
            fde_ =min(fde_,fd)
            ade_ls.append(ade_)
            fde_ls.append(fde_)
            print("ADE:",ade_," FDE:",fde_)

        print("*"*50)
        print("Avg ADE:",sum(ade_ls)/len(ade_ls) if ade_ls else 0)
        print("Avg FDE:",sum(fde_ls)/len(fde_ls) if fde_ls else 0)


if __name__ == '__main__':
    # ← 追加：argparseの設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='',
                        help='path to the saved model')
    parser.add_argument('--visualize', action='store_true',
                        help='flag to visualize the scenes')
    args = parser.parse_args()
    main(args)
