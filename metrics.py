import math
import numpy as np

# ADE (Average Displacement Error) の計算
def ade(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0
    num_samples_processed = 0 # 実際に処理されたサンプルの数をカウント

    for s in range(All):
        # predAll[s]は (pred_seq_len, num_peds, 2) の形状
        # targetAll[s]も (pred_seq_len, num_peds, 2) の形状
        # np.swapaxes(..., 0, 1) で (num_peds, pred_seq_len, 2) に変換
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)

        N = pred.shape[0] # 歩行者数
        # predとtargetのシーケンス長が異なる場合に備え、最小値を使用
        T = min(pred.shape[1], target.shape[1])

        # 歩行者数またはシーケンス長が0の場合はスキップ
        if N == 0 or T == 0:
            continue
        
        num_samples_processed += 1 # 処理されたサンプルをカウント

        sum_ = 0
        for i in range(N): # 各歩行者について
            for t in range(T): # 各タイムステップについて
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N*T) # 各サンプルでの平均ADE

    return sum_all/num_samples_processed if num_samples_processed > 0 else 0.0 # 全サンプルでの平均ADE

# FDE (Final Displacement Error) の計算
def fde(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0
    num_samples_processed = 0 # 実際に処理されたサンプルの数をカウント

    for s in range(All):
        # predAll[s]は (pred_seq_len, num_peds, 2) の形状
        # targetAll[s]も (pred_seq_len, num_peds, 2) の形状
        # np.swapaxes(..., 0, 1) で (num_peds, pred_seq_len, 2) に変換
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        N = pred.shape[0] # 歩行者数

        # predとtargetのシーケンス長が異なる場合に備え、最小値を使用
        T_pred = pred.shape[1]
        T_target = target.shape[1]

        # 歩行者数またはシーケンス長が0の場合はスキップ
        if N == 0 or T_pred == 0 or T_target == 0:
            continue

        num_samples_processed += 1 # 処理されたサンプルをカウント

        # 最終タイムステップのインデックスは、最小のシーケンス長から1を引いたもの
        final_t_idx = min(T_pred, T_target) - 1

        sum_ = 0
        for i in range(N): # 各歩行者について
            # 最終タイムステップについてのみ
            sum_+=math.sqrt((pred[i,final_t_idx,0] - target[i,final_t_idx,0])**2+(pred[i,final_t_idx,1] - target[i,final_t_idx,1])**2)
        sum_all += sum_/(N) # 各サンプルでの平均FDE

    return sum_all/num_samples_processed if num_samples_processed > 0 else 0.0 # 全サンプルでの平均FDE

# 以下は元のmetrics.pyに含まれる他の関数（変更なし）
# シーケンスデータをノード形式に変換
def seq_to_nodes(seq_):
    max_nodes = seq_.shape[1]
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
    nodes_ = np.zeros_like(nodes)
    num_peds_nodes = nodes.shape[1]
    num_peds_init_node = init_node.shape[0]
    num_peds_to_process = min(num_peds_nodes, num_peds_init_node)

    for s in range(nodes.shape[0]):
        for ped in range(num_peds_to_process):
            nodes_[s,ped,:] = np.sum(nodes[:s+1,ped,:],axis=0) + init_node[ped,:]
    return nodes_.squeeze()

# ゼロに近いかどうかを判定
def closer_to_zero(current,new_v):
    dec =  min([(abs(current),current),(abs(new_v),new_v)])[1]
    if dec != current:
        return True
    else:
        return False

# Bivariate Gaussian NLL損失の計算
def bivariate_loss(V_pred, V_trgt):
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
    epsilon = 1e-20
    result = result / denom
    result = -torch.log(torch.clamp(result, min=epsilon))
    return torch.mean(result)

# graph_loss関数
def graph_loss(V_pred, V_trgt):
    return bivariate_loss(V_pred, V_trgt)
   