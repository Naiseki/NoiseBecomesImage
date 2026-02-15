"""
粒子システム
マルチスケールスコア場を参照して粒子の位置と色を更新する
ステップの進行に応じてσを大→小へ連続的に切り替え、
大域構造から細部へと再構成を進める
元画像への直接参照は禁止
"""

import numpy as np
from scipy.ndimage import map_coordinates
from dataclasses import dataclass
from src.score_field import MultiScaleScoreField


@dataclass
class ParticleSystemConfig:
    """粒子システムの設定"""
    particle_count: int
    step_size: float          # η - 位置更新のステップサイズ
    color_step: float         # α - 色更新のステップサイズ
    color_sharpness: float    # β - 色の鮮やかさパラメータ
    noise_strength: float     # σ0 - 初期ノイズ強度（ランジュバンノイズ）
    total_steps: int          # T - 総ステップ数
    mode: str                 # "stochastic" or "deterministic"


class ParticleSystem:
    """マルチスケール粒子システムの管理クラス"""

    def __init__(
        self,
        config: ParticleSystemConfig,
        score_field: MultiScaleScoreField,
        seed: int | None = None
    ):
        """
        粒子システムを初期化

        Args:
            config: システム設定
            score_field: 事前計算されたマルチスケールスコア場
            seed: 乱数シード
        """
        self.config = config
        self.score_field = score_field
        self.rng = np.random.default_rng(seed)

        # 粒子の状態
        self.positions: np.ndarray  # (N, 2) - float座標 (x, y)
        self.colors: np.ndarray     # (N, 3) - RGB値 [0, 1]
        self.current_step: int = 0

        # 粒子を初期化
        self._initialize_particles()

    def _initialize_particles(self) -> None:
        """粒子を一様ランダムに初期化"""
        # 位置: [0, width) x [0, height) の範囲で一様ランダム
        self.positions = self.rng.random((self.config.particle_count, 2))
        self.positions[:, 0] *= self.score_field.width   # x座標
        self.positions[:, 1] *= self.score_field.height  # y座標

        # 色: [0, 1]の範囲で一様ランダム
        self.colors = self.rng.random((self.config.particle_count, 3))

    def _get_current_blur_sigma(self) -> float:
        """
        現在のステップに対応するぼかしσを計算
        σ_maxからσ_minへ等比的にアニーリング

        Returns:
            現在のぼかしσ
        """
        progress = self.current_step / max(self.config.total_steps, 1)
        sigma_max = self.score_field.sigma_levels[0]
        sigma_min = self.score_field.sigma_levels[-1]
        # 対数空間で線形補間 = 実空間で等比補間
        return sigma_max * (sigma_min / sigma_max) ** progress

    def _find_scale_weights(self, sigma: float) -> tuple[int, int, float]:
        """
        σに対応する2つの隣接スケールインデックスと補間重みを計算
        sigma_levelsは大→小の降順

        Args:
            sigma: 現在のぼかしσ

        Returns:
            (idx_lo, idx_hi, t) - idx_loが大きいσ側、idx_hiが小さいσ側
            tは0のときidx_lo、1のときidx_hiの値を使用
        """
        levels = self.score_field.sigma_levels

        # 範囲外のチェック
        if sigma >= levels[0]:
            return 0, 0, 0.0
        if sigma <= levels[-1]:
            n = len(levels) - 1
            return n, n, 0.0

        # 隣接する2つのレベルを探索（levelsは降順）
        for i in range(len(levels) - 1):
            if levels[i] >= sigma >= levels[i + 1]:
                # 対数空間で補間重みを計算
                log_ratio = np.log(levels[i] / sigma)
                log_span = np.log(levels[i] / levels[i + 1])
                t = log_ratio / log_span
                return i, i + 1, t

        # フォールバック
        return 0, 0, 0.0

    def _interpolate_field_2d(
        self,
        field: np.ndarray,
        positions: np.ndarray
    ) -> np.ndarray:
        """
        2Dフィールドに対してバイリニア補間

        Args:
            field: (H, W) のスカラーフィールド
            positions: (N, 2) 粒子位置 (x, y)

        Returns:
            (N,) 補間された値
        """
        # map_coordinatesは (y, x) の順番で座標を受け取る
        coords = np.array([positions[:, 1], positions[:, 0]])
        return map_coordinates(field, coords, order=1, mode='nearest')

    def _interpolate_score_at_level(
        self,
        positions: np.ndarray,
        level_idx: int
    ) -> np.ndarray:
        """
        指定スケールレベルでの位置スコアをバイリニア補間

        Args:
            positions: (N, 2) 粒子位置 (x, y)
            level_idx: スケールレベルのインデックス

        Returns:
            (N, 2) 補間されたスコア値 (∇x, ∇y)
        """
        score_field = self.score_field.score_pos_scales[level_idx]
        score_x = self._interpolate_field_2d(score_field[:, :, 0], positions)
        score_y = self._interpolate_field_2d(score_field[:, :, 1], positions)
        return np.stack([score_x, score_y], axis=-1)

    def _interpolate_multiscale_score(
        self,
        positions: np.ndarray,
        sigma: float
    ) -> np.ndarray:
        """
        現在のσに対応するスコアを隣接2レベルから補間して取得

        Args:
            positions: (N, 2) 粒子位置
            sigma: 現在のぼかしσ

        Returns:
            (N, 2) 補間されたスコア値
        """
        idx_lo, idx_hi, t = self._find_scale_weights(sigma)

        score_lo = self._interpolate_score_at_level(positions, idx_lo)
        if idx_lo == idx_hi:
            return score_lo

        score_hi = self._interpolate_score_at_level(positions, idx_hi)
        return (1.0 - t) * score_lo + t * score_hi

    def _compute_target_colors_at_level(
        self,
        positions: np.ndarray,
        level_idx: int
    ) -> np.ndarray:
        """
        指定スケールレベルでの目標色を計算
        softmax(β * [L_r, L_g, L_b]) で計算

        Args:
            positions: (N, 2) 粒子位置
            level_idx: スケールレベルのインデックス

        Returns:
            (N, 3) 目標色
        """
        L_r = self._interpolate_field_2d(
            self.score_field.log_density_r_scales[level_idx], positions
        )
        L_g = self._interpolate_field_2d(
            self.score_field.log_density_g_scales[level_idx], positions
        )
        L_b = self._interpolate_field_2d(
            self.score_field.log_density_b_scales[level_idx], positions
        )

        β = self.config.color_sharpness

        # (N, 3) にスタック
        log_densities = np.stack([L_r, L_g, L_b], axis=-1)

        # 数値安定性のため最大値を引く
        max_val = np.max(β * log_densities, axis=-1, keepdims=True)
        exp_values = np.exp(β * log_densities - max_val)

        # 正規化
        sum_exp = np.sum(exp_values, axis=-1, keepdims=True)
        return exp_values / sum_exp

    def _compute_target_colors_multiscale(
        self,
        positions: np.ndarray,
        sigma: float
    ) -> np.ndarray:
        """
        現在のσに対応する目標色を隣接2レベルから補間して取得

        Args:
            positions: (N, 2) 粒子位置
            sigma: 現在のぼかしσ

        Returns:
            (N, 3) 補間された目標色
        """
        idx_lo, idx_hi, t = self._find_scale_weights(sigma)

        colors_lo = self._compute_target_colors_at_level(positions, idx_lo)
        if idx_lo == idx_hi:
            return colors_lo

        colors_hi = self._compute_target_colors_at_level(positions, idx_hi)
        return (1.0 - t) * colors_lo + t * colors_hi

    def _compute_noise_scale(self) -> float:
        """
        ランジュバンノイズスケジュールの計算
        σ_noise(t) = σ0 * (1 - t/T)

        Returns:
            現在のノイズスケール
        """
        progress = self.current_step / max(self.config.total_steps, 1)
        return self.config.noise_strength * (1.0 - progress)

    def step(self) -> None:
        """
        1ステップの更新を実行

        1. 現在のぼかしσを計算（大→小へアニーリング）
        2. 対応するマルチスケールスコアで位置を更新
        3. 対応するマルチスケール目標色で色を更新
        """
        # 現在のぼかしσを計算
        blur_sigma = self._get_current_blur_sigma()

        # マルチスケールスコアから位置更新
        score = self._interpolate_multiscale_score(self.positions, blur_sigma)

        # ランジュバンノイズ項の計算
        if self.config.mode == "stochastic":
            noise_scale = self._compute_noise_scale()
            noise = noise_scale * self.rng.standard_normal(self.positions.shape)
        else:  # deterministic (DDIM)
            noise = 0.0

        # 位置を更新
        self.positions = self.positions + self.config.step_size * score + noise

        # 境界にクリップ
        self._clip_positions()

        # マルチスケール目標色で色を更新
        target_colors = self._compute_target_colors_multiscale(
            self.positions, blur_sigma
        )
        self.colors = self.colors + self.config.color_step * (target_colors - self.colors)

        # 色を [0, 1] にクリップ
        self.colors = np.clip(self.colors, 0.0, 1.0)

        # ステップ数を更新
        self.current_step += 1

    def _clip_positions(self) -> None:
        """位置を画像範囲内にクリップ"""
        self.positions[:, 0] = np.clip(
            self.positions[:, 0], 0, self.score_field.width - 1
        )
        self.positions[:, 1] = np.clip(
            self.positions[:, 1], 0, self.score_field.height - 1
        )

    def get_state(self) -> dict:
        """
        現在の状態を取得（描画用）

        Returns:
            位置、色、ステップ数、現在のσを含む辞書
        """
        return {
            "positions": self.positions.copy(),
            "colors": self.colors.copy(),
            "step": self.current_step,
            "blur_sigma": self._get_current_blur_sigma(),
        }
