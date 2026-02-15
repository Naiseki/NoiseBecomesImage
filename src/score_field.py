"""
マルチスケールスコア場の計算
画像から複数スケールの密度スコア場と色スコア場を事前計算する

各σについて:
  p_σ = gaussian_blur(p, σ)
  L_σ = log(p_σ)
  score_σ = ∇L_σ

サンプリング中はσを大→小へ連続的に切り替え、
大きいσで大域構造を形成し、小さいσで細部を再構成する
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass


@dataclass
class MultiScaleScoreField:
    """マルチスケールスコア場を保持するデータクラス"""
    sigma_levels: np.ndarray               # (L,) σ値（大→小の順）
    score_pos_scales: list[np.ndarray]     # L個の (H, W, 2) 密度スコア場
    log_density_r_scales: list[np.ndarray]  # L個の (H, W) 赤チャンネルlog密度
    log_density_g_scales: list[np.ndarray]  # L個の (H, W) 緑チャンネルlog密度
    log_density_b_scales: list[np.ndarray]  # L個の (H, W) 青チャンネルlog密度
    width: int
    height: int


def _compute_probability_density(
    channel: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    チャンネルからベース確率密度を計算

    Args:
        channel: 画像チャンネル (H, W)、値の範囲は [0, 1]
        epsilon: ゼロ除算/log(0)を避けるための微小値

    Returns:
        確率密度 (H, W)
    """
    channel_with_epsilon = channel + epsilon
    return channel_with_epsilon / np.sum(channel_with_epsilon)


def _compute_blurred_log_density(
    prob_density: np.ndarray,
    sigma: float,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    確率密度をσでぼかしてからlog変換

    Args:
        prob_density: ベース確率密度 (H, W)
        sigma: ガウシアンぼかしの標準偏差
        epsilon: log(0)を避けるための微小値

    Returns:
        ぼかし後のlog確率密度 (H, W)
    """
    blurred = gaussian_filter(prob_density, sigma=sigma)
    return np.log(blurred + epsilon)


def _compute_score_from_log_density(log_density: np.ndarray) -> np.ndarray:
    """
    log確率密度から勾配（スコア）を計算
    ぼかし済みなので追加のスムージングは不要

    Args:
        log_density: log確率密度 (H, W)

    Returns:
        勾配場 (H, W, 2) - (∇x, ∇y)
    """
    # np.gradientは (行方向, 列方向) = (y, x) の順番で返す
    grad_y, grad_x = np.gradient(log_density)
    return np.stack([grad_x, grad_y], axis=-1)


def build_multiscale_score_field(
    image: np.ndarray,
    sigma_max: float = 20.0,
    sigma_min: float = 0.5,
    num_scales: int = 10
) -> MultiScaleScoreField:
    """
    画像からマルチスケールスコア場を構築する（前処理）

    σ_maxからσ_minまでの等比数列でスケールを生成し、
    各スケールでぼかした確率密度からスコア場を計算

    Args:
        image: RGB画像 (H, W, 3)、値の範囲は [0, 1]
        sigma_max: 最大ぼかしσ（大域構造用）
        sigma_min: 最小ぼかしσ（細部用）
        num_scales: スケール数

    Returns:
        MultiScaleScoreField オブジェクト
    """
    height, width = image.shape[:2]

    # 輝度の計算: Y = 0.299R + 0.587G + 0.114B
    luminance = (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    )

    # ベース確率密度の計算
    prob_luminance = _compute_probability_density(luminance)
    prob_r = _compute_probability_density(image[:, :, 0])
    prob_g = _compute_probability_density(image[:, :, 1])
    prob_b = _compute_probability_density(image[:, :, 2])

    # σレベルの生成（大 → 小、等比数列）
    sigma_levels = np.geomspace(sigma_max, sigma_min, num_scales)

    # 各スケールでスコア場を事前計算
    score_pos_scales: list[np.ndarray] = []
    log_density_r_scales: list[np.ndarray] = []
    log_density_g_scales: list[np.ndarray] = []
    log_density_b_scales: list[np.ndarray] = []

    for sigma in sigma_levels:
        # 輝度密度をσでぼかしてスコア場を計算
        log_lum = _compute_blurred_log_density(prob_luminance, sigma)
        score = _compute_score_from_log_density(log_lum)
        score_pos_scales.append(score)

        # 各色チャンネルのlog密度もσでぼかして計算
        log_r = _compute_blurred_log_density(prob_r, sigma)
        log_g = _compute_blurred_log_density(prob_g, sigma)
        log_b = _compute_blurred_log_density(prob_b, sigma)

        log_density_r_scales.append(log_r)
        log_density_g_scales.append(log_g)
        log_density_b_scales.append(log_b)

    return MultiScaleScoreField(
        sigma_levels=sigma_levels,
        score_pos_scales=score_pos_scales,
        log_density_r_scales=log_density_r_scales,
        log_density_g_scales=log_density_g_scales,
        log_density_b_scales=log_density_b_scales,
        width=width,
        height=height,
    )
