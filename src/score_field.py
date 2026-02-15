"""
スコア場の計算
画像から密度スコア場と色スコア場を事前計算する
サンプリング中はこのスコア場のみを参照し、元画像は直接参照しない
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass


@dataclass
class ScoreField:
    """スコア場を保持するデータクラス"""
    score_pos: np.ndarray      # 密度スコア場 (H, W, 2) - (∇x, ∇y)
    log_density_r: np.ndarray  # 赤チャンネルのlog密度 (H, W)
    log_density_g: np.ndarray  # 緑チャンネルのlog密度 (H, W)
    log_density_b: np.ndarray  # 青チャンネルのlog密度 (H, W)
    width: int
    height: int


def compute_log_probability(channel: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    チャンネルの確率密度のlog値を計算

    Args:
        channel: 画像チャンネル (H, W)、値の範囲は [0, 1]
        epsilon: log(0)を避けるための微小値

    Returns:
        log確率密度 (H, W)
    """
    # 確率密度の計算
    # p(x,y) = (channel + ε) / Σ(channel + ε)
    channel_with_epsilon = channel + epsilon
    prob_density = channel_with_epsilon / np.sum(channel_with_epsilon)

    # log確率
    log_prob = np.log(prob_density)

    return log_prob


def compute_gradient_field(log_prob: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    ガウシアン微分を使って勾配場を計算

    Args:
        log_prob: log確率密度 (H, W)
        sigma: ガウシアンフィルタの標準偏差

    Returns:
        勾配場 (H, W, 2) - (∇x, ∇y)
    """
    # まずガウシアンぼかしを適用
    smoothed = gaussian_filter(log_prob, sigma=sigma)

    # 勾配を計算（中心差分）
    # np.gradientは (行方向, 列方向) = (y, x) の順番で返す
    grad_y, grad_x = np.gradient(smoothed)

    # (H, W, 2) の形式で返す: (∇x, ∇y)
    gradient_field = np.stack([grad_x, grad_y], axis=-1)

    return gradient_field


def build_score_field(image: np.ndarray, gradient_sigma: float = 1.0) -> ScoreField:
    """
    画像からスコア場を構築する（前処理）

    Args:
        image: RGB画像 (H, W, 3)、値の範囲は [0, 1]
        gradient_sigma: 勾配計算時のガウシアンぼかしの標準偏差

    Returns:
        ScoreField オブジェクト
    """
    height, width = image.shape[:2]

    # 輝度の計算
    # Y = 0.299R + 0.587G + 0.114B
    luminance = (
        0.299 * image[:, :, 0] +
        0.587 * image[:, :, 1] +
        0.114 * image[:, :, 2]
    )

    # 輝度からlog確率密度を計算
    log_luminance = compute_log_probability(luminance)

    # 密度スコア場を計算
    score_pos = compute_gradient_field(log_luminance, sigma=gradient_sigma)

    # 各RGBチャンネルのlog密度を計算
    log_density_r = compute_log_probability(image[:, :, 0])
    log_density_g = compute_log_probability(image[:, :, 1])
    log_density_b = compute_log_probability(image[:, :, 2])

    return ScoreField(
        score_pos=score_pos,
        log_density_r=log_density_r,
        log_density_g=log_density_g,
        log_density_b=log_density_b,
        width=width,
        height=height
    )
