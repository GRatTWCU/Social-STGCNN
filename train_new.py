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

def graph_loss(V_pred, V_target):
    return bivariate_loss(V_pred, V_target)

def calculate_ade_fde(V_pred, V_target, obs_traj, pred_traj_gt):
    """
    Calculate Average Displacement Error (ADE) and Final Displacement Error (FDE)
    
    Args:
        V_pred: Predicted trajectories [batch, seq_len, num_nodes, 2]
        V_target: Ground truth trajectories [batch, seq_len, num_nodes, 2]
        obs_traj: Observed trajectories [batch, obs_len, num_nodes, 2] 
        pred_traj_gt: Ground truth prediction trajectories [batch, pred_len, num_nodes, 2]
    
    Returns:
        ade: Average Displacement Error
        fde: Final Displacement Error
    """
    # Convert relative predictions to absolute coordinates
    # V_pred contains relative displacements, need to convert to absolute positions
    seq_len, num_nodes = V_pred.shape[1], V_pred.shape[2]
    
    # Get the last observed position for each pedestrian
    last_obs_pos = obs_traj[:, -1:, :, :]  # [batch, 1, num_nodes, 2]
    
    # Convert relative predictions to absolute positions
    pred_abs = torch.zeros_like(V_pred)
    pred_abs[:, 0:1, :, :] = last_obs_pos + V_pred[:, 0:1, :, :]
    
    for t in range(1, seq_len):
        pred_abs[:, t:t+1, :, :] = pred_abs[:, t-1:t, :, :] + V_pred[:, t:t+1, :, :]
    
    # Calculate displacement errors
    displacement_errors = torch.sqrt(torch.sum((pred_abs - pred_traj_gt) ** 2, dim=3))  # [batch, seq_len, num_nodes]
    
    # ADE: Average over all time steps and pedestrians
    ade = torch.mean(displacement_errors)
    
    # FDE: Error at the final time step, averaged over pedestrians
    fde = torch.mean(displacement_errors[:, -1, :])
    
    return ade.item(), fde.item()

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

        # Get data
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask, V_obs, A_obs, V_tr, A_tr = batch

        optimizer.zero_grad()
        
        # Forward
        # V_obs = batch,seq,node,feat
        # V_obs_tmp = batch,feat,seq,node
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
        
        V_pred = V_pred.permute(0, 2, 3, 1)
        
        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = graph_loss(V_pred, V_tr)
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
            
            # Calculate metrics
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

            # Get data
            batch = [tensor.to(device) for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
             loss_mask, V_obs, A_obs, V_tr, A_tr = batch
            
            V_obs_tmp = V_obs.permute(0, 3, 1, 2)

            V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
            
            V_pred = V_pred.permute(0, 2, 3, 1)
            
            V_tr = V_tr.squeeze()
            A_tr = A_tr.squeeze()
            V_pred = V_pred.squeeze()

            if batch_count % args.batch_size != 0 and cnt != turn_point:
                l = graph_loss(V_pred, V_tr)
                if is_fst_loss:
                    loss = l
                    is_fst_loss = False
                else:
                    loss += l
            else:
                loss = loss / args.batch_size
                is_fst_loss = True
                
                # Calculate metrics
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