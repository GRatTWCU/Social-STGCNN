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
# 変更: visualize.pyから描画関数をインポートします
from visualize import show_predictions

def test(args, KSTEPS=20):
    global loader_test, model
    # デバッグメッセージ: test関数が開始されたことを示します
    print("--- test function started ---")
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

        # --- デバッグメッセージ: 可視化フラグの状態を確認します ---
        print(f"--- [Scene {step}] Checking visualization flag: {args.visualize}")
        
        if args.visualize:
            # --- デバッグメッセージ: 可視化処理を開始することを示します ---
            print(f"--- [Scene {step}] Starting visualization process...")
            
            obs_traj_to_plot = raw_data_dict[step]['obs']
            gt_traj_to_plot = raw_data_dict[step]['trgt']
            all_pred_trajs_to_plot = raw_data_dict[step]['pred']
            
            # 保存先フォルダを準備します
            model_name = os.path.basename(os.path.dirname(args.model_path))
            save_dir = os.path.join("visualizations_output", model_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_file_path = os.path.join(save_dir, f"scene_{step:04d}.png")
            
            # visualize.pyの関数を呼び出して描画します
            show_predictions(obs_traj_to_plot, gt_traj_to_plot, all_pred_trajs_to_plot, save_file_path)
            
            # 可視化しすぎないように、一定数でループを抜けます
            if step > 50:
                print("--- Generated 50 visualization images. Exiting test loop. ---")
                break

    ade_ = sum(ade_bigls)/len(ade_bigls)
    fde_ = sum(fde_bigls)/len(fde_bigls)
    return ade_,fde_,raw_data_dict


def main(args):
    # デバッグメッセージ: main関数が開始されたことを示します
    print("--- main function started ---")
    paths = ['./checkpoint/*social-stgcnn*']
    KSTEPS=20

    print("*"*50)
    print('Number of samples:',KSTEPS)
    print("*"*50)

    # コマンドラインから--visualizeが指定されたか確認
    if args.visualize:
        print("--- Visualization is ENABLED ---")
    else:
        print("--- Visualization is DISABLED ---")
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

            if args.model_path != "":
                model_path = args.model_path

            # 可視化のために、現在のモデルパスをargsに保存します
            args.model_path = model_path

            with open(args_path,'rb') as f: 
                args_saved = pickle.load(f)

            stats= exp_path+'/constant_metrics.pkl'
            with open(stats,'rb') as f: 
                cm = pickle.load(f)
            print("Stats:",cm)
     
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
                    batch_size=1,
                    shuffle =False,
                    num_workers=1)

            model = social_stgcnn(n_stgcnn =args_saved.n_stgcnn,n_txpcnn=args_saved.n_txpcnn,
            output_feat=args_saved.output_size,seq_len=args_saved.obs_seq_len,
            kernel_size=args_saved.kernel_size,pred_seq_len=args_saved.pred_seq_len).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))

            ade_ =999999
            fde_ =999999
            print("Testing ....")
            ad,fd,raw_data_dic_= test(args, KSTEPS=KSTEPS)
            ade_= min(ade_,ad)
            fde_ =min(fde_,fd)
            ade_ls.append(ade_)
            fde_ls.append(fde_)
            print("ADE:",ade_," FDE:",fde_)

        print("*"*50)
        print("Avg ADE:",sum(ade_ls)/len(ade_ls) if ade_ls else 0)
        print("Avg FDE:",sum(fde_ls)/len(fde_ls) if fde_ls else 0)


if __name__ == '__main__':
    # コマンドライン引数を設定します
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='',
                        help='path to the saved model')
    parser.add_argument('--visualize', action='store_true',
                        help='flag to visualize the scenes')
    args = parser.parse_args()
    main(args)
