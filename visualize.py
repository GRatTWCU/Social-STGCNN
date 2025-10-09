import matplotlib
matplotlib.use('Agg')  # 非GUIバックエンドを指定
import matplotlib.pyplot as plt
import os
import numpy as np
import imageio
import glob

def show_predictions(obs_traj, pred_traj_gt, pred_trajs_all, save_path, xlim=None, ylim=None):
    """
    社会的相互作用を考慮した軌道を可視化し、画像として保存する関数。
    LRP風のビジュアルを、近接度に基づいた重み付けで再現します。
    """
    print(f"      -> Creating social visualization for: {os.path.basename(save_path)}")
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        num_peds = obs_traj.shape[1]

        if num_peds == 0:
            print(f"      ⚠️ No pedestrians to visualize in {os.path.basename(save_path)}. Skipping.")
            return

        # --- プライマリの人物を基準とした重みの計算 ---
        primary_ped_id = 0  # 最初の人物をプライマリとする

        # 全員の最後の観測位置を取得
        last_positions = obs_traj[-1, :, :]
        primary_pos = last_positions[primary_ped_id]

        # プライマリとの距離を計算
        distances = np.linalg.norm(last_positions - primary_pos, axis=1)
        
        # 距離の逆数を「影響度（重み）」とする (近いほど重みが大きい)
        # ゼロ除算を避けるために微小値を追加
        weights = 1.0 / (distances + 1e-6)
        weights[primary_ped_id] = 0  # 自分自身への重みは0

        # 重みを[0.1, 1.0]の範囲に正規化して、アルファ値（透明度）として利用
        max_weight = np.max(weights)
        if max_weight > 0:
            normalized_weights = 0.1 + 0.9 * (weights / max_weight)
        else:
            normalized_weights = np.zeros_like(weights)

        # --- 軌跡の描画 ---
        for i in range(num_peds):
            if i == primary_ped_id:
                # プライマリの人物の過去軌道 (青)
                ax.plot(obs_traj[:, i, 0], obs_traj[:, i, 1], 'b-', linewidth=2.5, label='Primary Pedestrian')
            else:
                # 他の人物の過去軌道 (オレンジ)
                ax.plot(obs_traj[:, i, 0], obs_traj[:, i, 1], color='orange', linewidth=2, label='Neighbor' if i == 1 else "")
                # 影響度を円の濃さ（アルファ値）で表現
                ax.scatter(obs_traj[:, i, 0], obs_traj[:, i, 1],
                           s=150,  # 円のサイズ
                           facecolors='orange',
                           alpha=normalized_weights[i],
                           edgecolors='none')

        # --- 予測の描画 (緑の矢印) ---
        # 予測サンプルの一つ目（どれでも良い）を矢印の描画に利用
        if pred_trajs_all:
            first_pred_sample = pred_trajs_all[0]
            for i in range(num_peds):
                start_pos = obs_traj[-1, i]
                # 予測軌道の最初のステップを終了点とする
                end_pos = first_pred_sample[0, i]
                dx = end_pos[0] - start_pos[0]
                dy = end_pos[1] - start_pos[1]
                ax.arrow(start_pos[0], start_pos[1], dx, dy,
                         head_width=0.15, head_length=0.15, fc='green', ec='green', length_includes_head=True,
                         label='Prediction' if i == 0 else "")

        ax.set_title(f"Social Interaction Visualization\n{os.path.basename(save_path)}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        # 凡例の重複をなくす
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        plt.savefig(save_path, dpi=150)
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
