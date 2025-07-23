import os
import math
import sys
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
from utils import * 
from metrics import * 
import pickle
import argparse
from torch import autograd
import torch.optim.lr_scheduler as lr_scheduler
from model import *

# 可視化のためのインポート
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns

# グラフ設定
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def graph_loss(V_pred,V_target):
    return bivariate_loss(V_pred,V_target)

class TrainingVisualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Training Progress Visualization', fontsize=16)
        
        # 損失プロット用の軸
        self.loss_ax = self.axes[0, 0]
        self.lr_ax = self.axes[0, 1]
        self.metrics_ax = self.axes[1, 0]
        self.stats_ax = self.axes[1, 1]
        
        # データ保存用リスト
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.learning_rates = []
        
    def update_plots(self, epoch, train_loss, val_loss, lr=None, metrics=None):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        if lr is not None:
            self.learning_rates.append(lr)
        
        # 損失プロットの更新
        self.loss_ax.clear()
        self.loss_ax.plot(self.epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        self.loss_ax.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.set_title('Training and Validation Loss')
        self.loss_ax.legend()
        self.loss_ax.grid(True, alpha=0.3)
        
        # 学習率プロット
        if self.learning_rates:
            self.lr_ax.clear()
            self.lr_ax.plot(self.epochs, self.learning_rates, 'g-', linewidth=2)
            self.lr_ax.set_xlabel('Epoch')
            self.lr_ax.set_ylabel('Learning Rate')
            self.lr_ax.set_title('Learning Rate Schedule')
            self.lr_ax.grid(True, alpha=0.3)
        
        # メトリクス表示
        self.metrics_ax.clear()
        if len(self.train_losses) > 1:
            # 最近の10エポックでの改善を表示
            recent_epochs = min(10, len(self.train_losses))
            recent_train = self.train_losses[-recent_epochs:]
            recent_val = self.val_losses[-recent_epochs:]
            recent_ep = self.epochs[-recent_epochs:]
            
            self.metrics_ax.plot(recent_ep, recent_train, 'b-', label='Train', linewidth=2)
            self.metrics_ax.plot(recent_ep, recent_val, 'r-', label='Val', linewidth=2)
            self.metrics_ax.set_title('Recent 10 Epochs')
            self.metrics_ax.legend()
            self.metrics_ax.grid(True, alpha=0.3)
        
        # 統計情報表示
        self.stats_ax.clear()
        self.stats_ax.axis('off')
        
        if self.train_losses:
            min_train_loss = min(self.train_losses)
            min_val_loss = min(self.val_losses)
            min_val_epoch = self.epochs[self.val_losses.index(min_val_loss)]
            
            stats_text = f"""
            Current Epoch: {epoch}
            Current Train Loss: {train_loss:.6f}
            Current Val Loss: {val_loss:.6f}
            
            Best Val Loss: {min_val_loss:.6f}
            Best Val Epoch: {min_val_epoch}
            
            Min Train Loss: {min_train_loss:.6f}
            """
            
            self.stats_ax.text(0.1, 0.9, stats_text, transform=self.stats_ax.transAxes, 
                             fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'training_progress_epoch_{epoch}.png'), 
                   dpi=100, bbox_inches='tight')
        plt.draw()
        plt.pause(0.1)
    
    def save_final_plot(self):
        plt.savefig(os.path.join(self.save_dir, 'final_training_progress.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()

def train(epoch, model, loader_train, optimizer, args, device):
    model.train()
    loss_batch = 0 
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    turn_point =int(loader_len/args.batch_size)*args.batch_size+ loader_len%args.batch_size -1

    for cnt,batch in enumerate(loader_train): 
        batch_count+=1

        #Get data
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask,V_obs,A_obs,V_tr,A_tr = batch

        optimizer.zero_grad()
        #Forward
        #V_obs = batch,seq,node,feat
        #V_obs_tmp = batch,feat,seq,node
        V_obs_tmp =V_obs.permute(0,3,1,2)

        V_pred,_ = model(V_obs_tmp,A_obs.squeeze())
        
        V_pred = V_pred.permute(0,2,3,1)
        
        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count%args.batch_size !=0 and cnt != turn_point :
            l = graph_loss(V_pred,V_tr)
            if is_fst_loss :
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss/args.batch_size
            is_fst_loss = True
            loss.backward()
            
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)

            optimizer.step()
            #Metrics
            loss_batch += loss.item()
            print('TRAIN:','\t Epoch:', epoch,'\t Loss:',loss_batch/batch_count)
            
    return loss_batch/batch_count

def vald(epoch, model, loader_val, args, device):
    model.eval()
    loss_batch = 0 
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_val)
    turn_point =int(loader_len/args.batch_size)*args.batch_size+ loader_len%args.batch_size -1
    
    for cnt,batch in enumerate(loader_val): 
        batch_count+=1

        #Get data
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask,V_obs,A_obs,V_tr,A_tr = batch
        
        V_obs_tmp =V_obs.permute(0,3,1,2)

        V_pred,_ = model(V_obs_tmp,A_obs.squeeze())
        
        V_pred = V_pred.permute(0,2,3,1)
        
        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count%args.batch_size !=0 and cnt != turn_point :
            l = graph_loss(V_pred,V_tr)
            if is_fst_loss :
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss/args.batch_size
            is_fst_loss = True
            #Metrics
            loss_batch += loss.item()
            print('VALD:','\t Epoch:', epoch,'\t Loss:',loss_batch/batch_count)

    return loss_batch/batch_count

def main():
    parser = argparse.ArgumentParser()

    #Model specific parameters
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    parser.add_argument('--n_stgcnn', type=int, default=1,help='Number of ST-GCNN layers')
    parser.add_argument('--n_txpcnn', type=int, default=5, help='Number of TXPCNN layers')
    parser.add_argument('--kernel_size', type=int, default=3)

    #Data specifc paremeters
    parser.add_argument('--obs_seq_len', type=int, default=8)
    parser.add_argument('--pred_seq_len', type=int, default=12)
    parser.add_argument('--dataset', default='eth',
                        help='eth,hotel,univ,zara1,zara2')    

    #Training specifc parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=250,
                        help='number of epochs')  
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='gadient clipping')        
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_sh_rate', type=int, default=150,
                        help='number of steps to drop the lr')  
    parser.add_argument('--use_lrschd', action="store_true", default=False,
                        help='Use lr rate scheduler')
    parser.add_argument('--tag', default='tag',
                        help='personal tag for the model ')
    
    # 可視化関連のパラメータ
    parser.add_argument('--plot_every', type=int, default=1,
                        help='Plot every N epochs')
    parser.add_argument('--save_plots', action="store_true", default=True,
                        help='Save training plots')
                        
    args = parser.parse_args()

    print('*'*30)
    print("Training initiating....")
    print(args)

    #Data prep     
    obs_seq_len = args.obs_seq_len
    pred_seq_len = args.pred_seq_len
    data_set = './datasets/'+args.dataset+'/'

    dset_train = TrajectoryDataset(
            data_set+'train/',
            obs_len=obs_seq_len,
            pred_len=pred_seq_len,
            skip=1,norm_lap_matr=True)

    loader_train = DataLoader(
            dset_train,
            batch_size=1, #This is irrelative to the args batch size parameter
            shuffle =True,
            num_workers=0)

    dset_val = TrajectoryDataset(
            data_set+'val/',
            obs_len=obs_seq_len,
            pred_len=pred_seq_len,
            skip=1,norm_lap_matr=True)

    loader_val = DataLoader(
            dset_val,
            batch_size=1, #This is irrelative to the args batch size parameter
            shuffle =False,
            num_workers=0)

    #Defining the model 
    model = social_stgcnn(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
    output_feat=args.output_size,seq_len=args.obs_seq_len,
    kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).to(device)

    #Training settings 
    optimizer = optim.SGD(model.parameters(),lr=args.lr)

    if args.use_lrschd:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)

    checkpoint_dir = './checkpoint/'+args.tag+'/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # 可視化用のディレクトリ作成
    plots_dir = os.path.join(checkpoint_dir, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    with open(checkpoint_dir+'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    print('Data and model loaded')
    print('Checkpoint dir:', checkpoint_dir)

    # 可視化器の初期化
    if args.save_plots:
        visualizer = TrainingVisualizer(plots_dir)
        plt.ion()  # インタラクティブモードをオン

    #Training 
    metrics = {'train_loss':[],  'val_loss':[]}
    constant_metrics = {'min_val_epoch':-1, 'min_val_loss':9999999999999999}

    print('Training started ...')
    for epoch in range(args.num_epochs):
        train_loss = train(epoch, model, loader_train, optimizer, args, device)
        val_loss = vald(epoch, model, loader_val, args, device)
        
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        
        # 学習率の取得
        current_lr = optimizer.param_groups[0]['lr']
        
        if args.use_lrschd:
            scheduler.step()

        if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
            constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
            constant_metrics['min_val_epoch'] = epoch
            torch.save(model.state_dict(),checkpoint_dir+'val_best.pth')

        print('*'*30)
        print('Epoch:',args.tag,":", epoch)
        for k,v in metrics.items():
            if len(v)>0:
                print(k,v[-1])

        print(constant_metrics)
        print('*'*30)
        
        # 可視化の更新
        if args.save_plots and epoch % args.plot_every == 0:
            visualizer.update_plots(epoch, train_loss, val_loss, current_lr, metrics)
        
        with open(checkpoint_dir+'metrics.pkl', 'wb') as fp:
            pickle.dump(metrics, fp)
        
        with open(checkpoint_dir+'constant_metrics.pkl', 'wb') as fp:
            pickle.dump(constant_metrics, fp)

    # 最終的なプロットの保存
    if args.save_plots:
        visualizer.save_final_plot()
        plt.ioff()  # インタラクティブモードをオフ

    print('Training completed!')
    print(f'Best validation loss: {constant_metrics["min_val_loss"]:.6f} at epoch {constant_metrics["min_val_epoch"]}')

if __name__ == '__main__':
    # マルチプロセシング対応
    import multiprocessing
    multiprocessing.freeze_support()
    main()