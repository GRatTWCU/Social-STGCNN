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
import shutil
try:
    # Google Colab環境でのみインポートし、ダウンロードに使用します
    from google.colab import files
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

from utils import *
from metrics import *
from model import social_stgcnn
import copy
# visualize.pyから2つの関数をインポートします
from visualize import show_predictions, create_gif

def test(model, loader_test, args, xlim=None, ylim=None, KSTEPS=20):
    """
    モデルのテストを実行する関数
    xlim, ylim: 可視化時の座標範囲
    """
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
        with torch.no_grad():
            V_pred,_ = model(V_obs_tmp,A_obs.squeeze())
        V_pred = V_pred.permute(0,2,3,1)

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred,V_tr = V_pred[:,:num_of_objs,:],V_tr[:,:num_of_objs,:]
        
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

        if args.visualize:
            model_name = os.path.basename(os.path.dirname(args.model_path))
            save_dir = os.path.join("visualizations_output", model_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_file_path = os.path.join(save_dir, f"scene_{step:04d}.png")
            
            # 修正：show_predictionsにxlimとylimを渡す
            show_predictions(
                raw_data_dict[step]['obs'],
                raw_data_dict[step]['trgt'],
                raw_data_dict[step]['pred'],
                save_file_path,
                xlim=xlim,
                ylim=ylim
            )
            
            # 可視化は50フレームまでで一度停止
            if step > 50:
                print("--- Generated 50 visualization images. Exiting test loop. ---")
                break

    ade_ = sum(ade_bigls)/len(ade_bigls) if ade_bigls else 0
    fde_ = sum(fde_bigls)/len(fde_bigls) if fde_bigls else 0
    return ade_,fde_,raw_data_dict


def main(args):
    print("--- main function started ---")
    paths = ['./checkpoint/*social-stgcnn*']
    KSTEPS=20

    print("*"*50)
    print('Number of samples:',KSTEPS)
    print("*"*50)

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
            
            # --- ▼▼▼ ここからが追加された処理です ▼▼▼ ---
            xlim = None
            ylim = None
            if args.visualize:
                # データセット全体の座標範囲を計算
                print("Calculating dataset bounds for consistent visualization...")
                all_obs_traj_list = []
                all_pred_traj_gt_list = []
                # DataLoaderをループして全データを集める
                for batch_data in loader_test:
                    obs_traj, pred_traj_gt, _, _, _, _, _, _, _, _ = batch_data
                    all_obs_traj_list.append(obs_traj)
                    all_pred_traj_gt_list.append(pred_traj_gt)
                
                # バッチの次元で連結
                all_obs_traj = torch.cat(all_obs_traj_list, dim=1).squeeze(0) # [seq, peds, xy]
                all_pred_traj_gt = torch.cat(all_pred_traj_gt_list, dim=1).squeeze(0) # [seq, peds, xy]

                # 観測と真値の両方の軌道データを結合
                full_traj = torch.cat([all_obs_traj, all_pred_traj_gt], dim=0)

                # 少し余白(padding)を持たせる
                padding = 2.0 
                xlim = (full_traj[:, :, 0].min().item() - padding, full_traj[:, :, 0].max().item() + padding)
                ylim = (full_traj[:, :, 1].min().item() - padding, full_traj[:, :, 1].max().item() + padding)
                print(f"Determined visualization bounds: xlim={xlim}, ylim={ylim}")
            # --- ▲▲▲ 追加された処理はここまで ▲▲▲ ---


            model = social_stgcnn(n_stgcnn =args_saved.n_stgcnn,n_txpcnn=args_saved.n_txpcnn,
            output_feat=args_saved.output_size,seq_len=args_saved.obs_seq_len,
            kernel_size=args_saved.kernel_size,pred_seq_len=args_saved.pred_seq_len).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))

            ade_ =999999
            fde_ =999999
            print("Testing ....")
            # 修正：test関数にxlimとylimを渡す
            ad,fd,raw_data_dic_= test(model, loader_test, args, xlim=xlim, ylim=ylim, KSTEPS=KSTEPS)
            ade_= min(ade_,ad)
            fde_ =min(fde_,fd)
            ade_ls.append(ade_)
            fde_ls.append(fde_)
            print("ADE:",ade_," FDE:",fde_)

            # --- ▼▼▼ ここからが追加された後処理です ▼▼▼ ---
            if args.visualize:
                model_name = os.path.basename(os.path.dirname(args.model_path))
                image_folder = os.path.join("visualizations_output", model_name)
                
                # 1. GIFを作成
                gif_path = f"visualizations_output/{model_name}_animation.gif"
                create_gif(image_folder, gif_path)

                # 2. PNG画像をZIPに圧縮
                zip_path_base = os.path.join("visualizations_output", f"{model_name}_images")
                zip_path = None
                try:
                    shutil.make_archive(zip_path_base, 'zip', image_folder)
                    zip_path = f"{zip_path_base}.zip"
                    print(f"      ✅ ZIP archive of images successfully saved to: {zip_path}")
                except Exception as e:
                    print(f"      ❌ FAILED to create ZIP archive. Error: {e}")

                # 3. Colab環境であればダウンロードをトリガー
                if IS_COLAB:
                    print(f"--- Triggering downloads for {model_name}. Please check your browser. ---")
                    try:
                        files.download(gif_path)
                        if zip_path:
                            files.download(zip_path)
                    except Exception as e:
                        print(f"      ❌ Could not trigger automatic download. Error: {e}")
                        print(f"      You can download the files manually from the file browser on the left.")
                else:
                    print(f"--- Find your generated files in the 'visualizations_output' directory. ---")
            # --- ▲▲▲ 追加された後処理はここまで ▲▲▲ ---

    print("*"*50)
    print("Avg ADE:",sum(ade_ls)/len(ade_ls) if ade_ls else 0)
    print("Avg FDE:",sum(fde_ls)/len(fde_ls) if fde_ls else 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='',
                        help='path to the saved model')
    parser.add_argument('--visualize', action='store_true',
                        help='flag to visualize the scenes')
    args = parser.parse_args()
    main(args)
