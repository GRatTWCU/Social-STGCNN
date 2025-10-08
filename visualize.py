import matplotlib
matplotlib.use('Agg')  # 非GUIバックエンドを指定
import matplotlib.pyplot as plt
import os
import numpy as np
import imageio
import glob

def show_predictions(obs_traj, pred_traj_gt, pred_trajs_all, save_path, xlim=None, ylim=None):
    """
    軌道を可視化して画像として保存する関数
    xlim, ylim: (min, max)のタプルで座標範囲を指定
    """
    # --- デバッグメッセージ ---
    print(f"      -> Visualizing scene: {os.path.basename(save_path)}")
    try:
        fig, ax = plt.subplots(figsize=(10, 10))

        num_peds = obs_traj.shape[1]

        for i in range(num_peds):
            # 過去の軌道を描画 (青の破線)
            ax.plot(obs_traj[:, i, 0], obs_traj[:, i, 1], 'b--')
            # 軌道の開始点
            ax.plot(obs_traj[0, i, 0], obs_traj[0, i, 1], 'bo', label='Start' if i == 0 else "")

            # 正解の未来軌道を描画 (緑の実線)
            # 観測の最終点と真値の未来軌道をつなげる
            full_gt_traj = np.concatenate([obs_traj[-1:, i, :], pred_traj_gt[:, i, :]])
            ax.plot(full_gt_traj[:, 0], full_gt_traj[:, 1], 'g-', linewidth=2, label='Ground Truth' if i == 0 else "")

        # 複数の予測軌道をすべて描画
        for k in range(len(pred_trajs_all)):
            pred_traj = pred_trajs_all[k]
            for i in range(num_peds):
                # 観測の最終点と予測軌道をつなげる
                full_pred_traj = np.concatenate([obs_traj[-1:, i, :], pred_traj[:, i, :]])
                # 透明度を下げて薄い赤線で描画
                ax.plot(full_pred_traj[:, 0], full_pred_traj[:, 1], 'r-', linewidth=1, alpha=0.3, label='Prediction' if i == 0 and k == 0 else "")

        ax.set_title(os.path.basename(save_path))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        ax.grid(True)
        
        # --- ▼▼▼ ここからが修正箇所です ▼▼▼ ---
        # 座標のアスペクト比を1:1に保つ
        ax.set_aspect('equal', adjustable='box')
        
        # 外部から指定された表示範囲があれば設定する
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        # --- ▲▲▲ 修正箇所はここまで ▲▲▲ ---

        # ファイルに保存
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        
        # --- デバッグメッセージ ---
        print(f"      ✅ Image successfully saved to: {save_path}")

    except Exception as e:
        # --- エラーメッセージ ---
        print(f"      ❌ FAILED to save image. Error: {e}")

def create_gif(image_folder, gif_path, duration=0.2):
    """
    指定されたフォルダ内のPNG画像からアニメーションGIFを作成する関数
    """
    print(f"--- Creating GIF from images in: {image_folder} ---")
    try:
        # PNGファイルを数字順に正しく並び替える
        files = glob.glob(os.path.join(image_folder, '*.png'))
        files.sort()

        if not files:
            print(f"      ⚠️ No images found in {image_folder}. Cannot create GIF.")
            return

        images = []
        for filename in files:
            images.append(imageio.imread(filename))
        
        imageio.mimsave(gif_path, images, duration=duration)
        print(f"      ✅ GIF successfully saved to: {gif_path}")
    except Exception as e:
        print(f"      ❌ FAILED to create GIF. Error: {e}")
