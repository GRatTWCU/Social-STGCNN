import matplotlib
matplotlib.use('Agg')  # 非GUIバックエンドを指定
import matplotlib.pyplot as plt
import os
import numpy as np
import imageio
import glob

# 日本語フォントを表示するための設定
# Colabや多くの環境で動作するIPAフォントを指定
plt.rcParams['font.sans-serif'] = ['IPAexGothic']
plt.rcParams['axes.unicode_minus'] = False # マイナス記号の文字化けを防ぐ

def show_predictions(obs_traj, pred_traj_gt, pred_trajs_all, save_path, xlim=None, ylim=None):
    """
    確率的な未来予測と正解軌道を比較して可視化する関数。
    凡例を日本語化し、分かりやすくしています。
    """
    print(f"      -> Creating visualization with Japanese legend for: {os.path.basename(save_path)}")
    try:
        fig, ax = plt.subplots(figsize=(12, 12)) # 少し大きめの図に変更
        num_peds = obs_traj.shape[1]

        if num_peds == 0:
            print(f"      ⚠️ No pedestrians to visualize in {os.path.basename(save_path)}. Skipping.")
            return

        # --- プライマリ人物を基準とした影響度の計算 ---
        primary_ped_id = 0
        last_positions = obs_traj[-1, :, :]
        primary_pos = last_positions[primary_ped_id]
        distances = np.linalg.norm(last_positions - primary_pos, axis=1)
        weights = 1.0 / (distances + 1e-6)
        weights[primary_ped_id] = 0
        max_weight = np.max(weights)
        if max_weight > 0:
            normalized_weights = 0.1 + 0.9 * (weights / max_weight)
        else:
            normalized_weights = np.zeros_like(weights)

        # --- 軌跡の描画 ---
        for i in range(num_peds):
            # --- 1. 過去の軌道 (観測) ---
            if i == primary_ped_id:
                ax.plot(obs_traj[:, i, 0], obs_traj[:, i, 1], 'b-', linewidth=2.5, label='予測対象（過去の軌道）')
            else:
                ax.plot(obs_traj[:, i, 0], obs_traj[:, i, 1], color='orange', linewidth=2, label='周囲の人物（過去の軌道）' if i == 1 else "")
                ax.scatter(obs_traj[:, i, 0], obs_traj[:, i, 1], s=150, facecolors='orange', alpha=normalized_weights[i], edgecolors='none')

            # --- 2. 正解の未来軌道 ---
            last_obs_pos = obs_traj[-1:, i, :]
            full_gt_traj = np.concatenate([last_obs_pos, pred_traj_gt[:, i, :]])
            ax.plot(full_gt_traj[:, 0], full_gt_traj[:, 1], 'g-', linewidth=2.5, label='正解の未来軌道' if i == 0 else "")

        # --- 3. 複数の予測軌道 ---
        if pred_trajs_all:
            # 全ての予測サンプルをループして描画
            for k in range(len(pred_trajs_all)):
                pred_traj_sample = pred_trajs_all[k]
                for i in range(num_peds):
                    last_obs_pos = obs_traj[-1:, i, :]
                    full_pred_traj = np.concatenate([last_obs_pos, pred_traj_sample[:, i, :]])
                    # 半透明の細い赤線で描画
                    ax.plot(full_pred_traj[:, 0], full_pred_traj[:, 1], 'r-', linewidth=1, alpha=0.2, label='予測された未来軌道' if i == 0 and k == 0 else "")

        ax.set_title(f"予測分布と正解軌道の比較\n{os.path.basename(save_path)}", fontsize=16)
        ax.set_xlabel("X座標", fontsize=12)
        ax.set_ylabel("Y座標", fontsize=12)
        
        # 凡例の重複をなくして表示
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=12)

        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"      ✅ Image successfully saved to: {save_path}")

    except Exception as e:
        print(f"      ❌ FAILED to save visualization. Error: {e}")


def create_gif(image_folder, gif_path, duration=0.2):
    """
    指定されたフォルダ内のPNG画像からアニメーションGIFを作成する関数
    """
    print(f"--- Creating GIF from images in: {image_folder} ---")
    try:
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
