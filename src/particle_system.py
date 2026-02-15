"""
粒子システム
スコア場のみを参照して粒子の位置と色を更新する
元画像への直接参照は禁止
"""

import numpy as np
from scipy.ndimage import map_coordinates
from dataclasses import dataclass
from src.score_field import ScoreField


@dataclass
class ParticleSystemConfig:
    """粒子システムの設定"""
    particle_count: int
    step_size: float          # η - 位置更新のステップサイズ
    color_step: float         # α - 色更新のステップサイズ
    color_sharpness: float    # β - 色の鮮やかさパラメータ
    noise_strength: float     # σ0 - 初期ノイズ強度
    total_steps: int          # T - 総ステップ数
    mode: str                 # "stochastic" or "deterministic"


class ParticleSystem:
    """粒子システムの管理クラス"""

    def __init__(
        self,
        config: ParticleSystemConfig,
        score_field: ScoreField,
        seed: int | None = None
    ):
        """
        粒子システムを初期化

        Args:
            config: システム設定
            score_field: 事前計算されたスコア場
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

    def _interpolate_score_pos(self, positions: np.ndarray) -> np.ndarray:
        """
        位置スコアをバイリニア補間

        Args:
            positions: (N, 2) 粒子位置 (x, y)

        Returns:
            (N, 2) 補間されたスコア値 (∇x, ∇y)
        """
        # map_coordinatesは (y, x) の順番で座標を受け取る
        # positionsは (x, y) なので、順番を入れ替える
        coords = np.array([positions[:, 1], positions[:, 0]])  # (2, N) - (y, x)

        # 各勾配成分を補間
        score_x = map_coordinates(
            self.score_field.score_pos[:, :, 0],  # ∇x成分
            coords,
            order=1,  # バイリニア補間
            mode='nearest'  # 境界処理
        )
        score_y = map_coordinates(
            self.score_field.score_pos[:, :, 1],  # ∇y成分
            coords,
            order=1,
            mode='nearest'
        )

        # (N, 2) の形式で返す
        return np.stack([score_x, score_y], axis=-1)

    def _interpolate_log_density(
        self,
        log_density: np.ndarray,
        positions: np.ndarray
    ) -> np.ndarray:
        """
        log密度値をバイリニア補間

        Args:
            log_density: (H, W) log密度フィールド
            positions: (N, 2) 粒子位置 (x, y)

        Returns:
            (N,) 補間されたlog密度値
        """
        # (y, x) の順番で座標を渡す
        coords = np.array([positions[:, 1], positions[:, 0]])

        values = map_coordinates(
            log_density,
            coords,
            order=1,
            mode='nearest'
        )

        return values

    def _compute_target_colors(self, positions: np.ndarray) -> np.ndarray:
        """
        現在位置での目標色を計算
        softmax(β * [L_r, L_g, L_b]) で計算

        Args:
            positions: (N, 2) 粒子位置

        Returns:
            (N, 3) 目標色
        """
        # 各チャンネルのlog密度を補間
        L_r = self._interpolate_log_density(
            self.score_field.log_density_r, positions
        )
        L_g = self._interpolate_log_density(
            self.score_field.log_density_g, positions
        )
        L_b = self._interpolate_log_density(
            self.score_field.log_density_b, positions
        )

        # Softmax で目標色を計算
        # μ_c = exp(β * L_c) / Σ_c exp(β * L_c)
        β = self.config.color_sharpness

        # (N, 3) の形式にスタック
        log_densities = np.stack([L_r, L_g, L_b], axis=-1)

        # exp(β * L)を計算
        exp_values = np.exp(β * log_densities)

        # 正規化
        sum_exp = np.sum(exp_values, axis=-1, keepdims=True)
        target_colors = exp_values / sum_exp

        return target_colors

    def _compute_noise_scale(self) -> float:
        """
        ノイズスケジュールの計算
        σ(t) = σ0 * (1 - t/T)

        Returns:
            現在のノイズスケール
        """
        progress = self.current_step / self.config.total_steps
        noise_scale = self.config.noise_strength * (1.0 - progress)
        return noise_scale

    def step(self) -> None:
        """
        1ステップの更新を実行

        1. 位置更新: x_{t+1} = x_t + η * score_pos(x_t) + σ(t) * N(0,1)
        2. 色更新: c_{t+1} = c_t + α * (μ(x_t) - c_t)
        3. 境界処理
        """
        # 位置更新
        score = self._interpolate_score_pos(self.positions)

        # ノイズ項の計算
        if self.config.mode == "stochastic":
            noise_scale = self._compute_noise_scale()
            noise = noise_scale * self.rng.standard_normal(self.positions.shape)
        else:  # deterministic (DDIM)
            noise = 0.0

        # 位置を更新
        self.positions = self.positions + self.config.step_size * score + noise

        # 境界にクリップ
        self._clip_positions()

        # 色更新
        target_colors = self._compute_target_colors(self.positions)
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
            位置、色、ステップ数を含む辞書
        """
        return {
            "positions": self.positions.copy(),
            "colors": self.colors.copy(),
            "step": self.current_step
        }
