import matplotlib
matplotlib.use('Agg')  # 追加: 非GUIバックエンドを指定
import matplotlib.pyplot as plt
import os
import numpy as np

def show_predictions(obs_traj, pred_traj_gt, pred_trajs_all, save_path):
    """
    軌道を可視化して画像として保存する関数
    """
    # --- デバッグメッセージ ---
    print(f"    -> Entering show_predictions function for: {os.path.basename(save_path)}")
    try:
        fig, ax = plt.subplots(figsize=(10, 10))

        num_peds = obs_traj.shape[1]

        for i in range(num_peds):
            # 過去の軌道を描画 (青の破線)
            ax.plot(obs_traj[:, i, 0], obs_traj[:, i, 1], 'b--')
            # 軌道の開始点
            ax.plot(obs_traj[0, i, 0], obs_traj[0, i, 1], 'bo', label='Start' if i == 0 else "")

            # 正解の未来軌道を描画 (緑の実線)
            full_gt_traj = np.concatenate([obs_traj[-1:, i, :], pred_traj_gt[:, i, :]])
            ax.plot(full_gt_traj[:, 0], full_gt_traj[:, 1], 'g-', linewidth=2, label='Ground Truth' if i == 0 else "")

        # 複数の予測軌道をすべて描画
        for k in range(len(pred_trajs_all)):
            pred_traj = pred_trajs_all[k]
            for i in range(num_peds):
                full_pred_traj = np.concatenate([obs_traj[-1:, i, :], pred_traj[:, i, :]])
                # 透明度を下げて薄い赤線で描画
                ax.plot(full_pred_traj[:, 0], full_pred_traj[:, 1], 'r-', linewidth=1, alpha=0.3, label='Prediction' if i == 0 and k == 0 else "")

        ax.set_title(os.path.basename(save_path))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        ax.grid(True)
        ax.axis('equal')

        # ファイルに保存
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        
        # --- デバッグメッセージ ---
        print(f"    ✅ Image successfully saved to: {save_path}")

    except Exception as e:
        # --- エラーメッセージ ---
        print(f"    ❌ FAILED to save image. Error: {e}")
