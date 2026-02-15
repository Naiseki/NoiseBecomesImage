"""
レンダリングとGIF生成
粒子を画像に描画し、GIFとして保存する
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import imageio


def render_particles(
    positions: np.ndarray,
    colors: np.ndarray,
    width: int,
    height: int,
    blur_sigma: float
) -> np.ndarray:
    """
    粒子を画像に描画

    プロセス:
    1. 各粒子の色を位置に加算
    2. ガウシアンぼかし適用
    3. 正規化

    Args:
        positions: (N, 2) 粒子位置 (x, y)
        colors: (N, 3) 粒子色 RGB [0, 1]
        width: 画像幅
        height: 画像高さ
        blur_sigma: ガウシアンぼかしの標準偏差

    Returns:
        (H, W, 3) 描画された画像、値の範囲は [0, 1]
    """
    # 累積用の配列
    accum_rgb = np.zeros((height, width, 3), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.float64)

    # 粒子位置を整数座標に変換
    coords = np.clip(positions.astype(int), [0, 0], [width - 1, height - 1])
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    # np.add.at を使って効率的に累積
    for c in range(3):
        np.add.at(accum_rgb[:, :, c], (y_coords, x_coords), colors[:, c])
    np.add.at(count, (y_coords, x_coords), 1)

    # ガウシアンぼかしを適用
    for c in range(3):
        accum_rgb[:, :, c] = gaussian_filter(accum_rgb[:, :, c], sigma=blur_sigma)
    count = gaussian_filter(count, sigma=blur_sigma)

    # 正規化（ゼロ除算を避ける）
    output = accum_rgb / np.maximum(count[..., None], 1e-8)

    # [0, 1]にクリップ
    output = np.clip(output, 0.0, 1.0)

    return output


class GifRecorder:
    """GIFの録画クラス"""

    def __init__(self, max_frames: int = 300):
        """
        GIFレコーダーを初期化

        Args:
            max_frames: 最大フレーム数
        """
        self.frames: list[np.ndarray] = []
        self.max_frames = max_frames

    def add_frame(self, image: np.ndarray) -> None:
        """
        フレームを追加

        Args:
            image: (H, W, 3) RGB画像、値の範囲は [0, 1]
        """
        # [0, 1] → [0, 255] uint8 に変換
        frame = (image * 255).astype(np.uint8)

        # 最大フレーム数以下の場合のみ追加
        if len(self.frames) < self.max_frames:
            self.frames.append(frame)

    def save_gif(self, filepath: str, fps: int = 20) -> None:
        """
        GIFとして保存

        Args:
            filepath: 保存先パス
            fps: フレームレート（frames per second）
        """
        if not self.frames:
            raise ValueError("保存するフレームがありません")

        # FPSから各フレームの表示時間を計算 (ミリ秒)
        duration = 1000 / fps

        # GIFとして保存
        imageio.mimsave(
            filepath,
            self.frames,
            duration=duration,
            loop=0  # 無限ループ
        )

    def clear(self) -> None:
        """フレームをクリア"""
        self.frames = []

    def frame_count(self) -> int:
        """現在のフレーム数を取得"""
        return len(self.frames)
