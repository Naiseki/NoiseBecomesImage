"""
Streamlit Webアプリケーション
Noise Becomes Image - スコアベース粒子システムによる画像再構成
"""

import streamlit as st
import numpy as np
from src.utils import load_and_preprocess_image
from src.score_field import build_score_field
from src.particle_system import ParticleSystem, ParticleSystemConfig
from src.renderer import render_particles, GifRecorder


def initialize_session_state() -> None:
    """セッション状態を初期化"""
    if "score_field" not in st.session_state:
        st.session_state.score_field = None
    if "particle_system" not in st.session_state:
        st.session_state.particle_system = None
    if "gif_recorder" not in st.session_state:
        st.session_state.gif_recorder = None
    if "original_image" not in st.session_state:
        st.session_state.original_image = None
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None


def render_sidebar() -> dict:
    """
    サイドバーのUIを描画し、パラメータを取得

    Returns:
        パラメータ辞書
    """
    st.sidebar.header("パラメータ設定")

    params = {}

    params["particle_count"] = st.sidebar.slider(
        "粒子数",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000,
        help="シミュレーションで使用する粒子の数"
    )

    params["step_size"] = st.sidebar.slider(
        "位置更新ステップ (η)",
        min_value=0.01,
        max_value=2.0,
        value=0.5,
        step=0.01,
        help="粒子の位置更新の大きさ"
    )

    params["color_step"] = st.sidebar.slider(
        "色更新ステップ (α)",
        min_value=0.01,
        max_value=1.0,
        value=0.2,
        step=0.01,
        help="粒子の色更新の速度"
    )

    params["color_sharpness"] = st.sidebar.slider(
        "色の鮮やかさ (β)",
        min_value=1.0,
        max_value=15.0,
        value=5.0,
        step=0.5,
        help="色の鮮やかさを制御するパラメータ（大きいほど鮮やか）"
    )

    params["noise_strength"] = st.sidebar.slider(
        "ノイズ強度 (σ0)",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="確率的モードでのノイズの強さ"
    )

    params["total_steps"] = st.sidebar.slider(
        "総ステップ数",
        min_value=50,
        max_value=1000,
        value=200,
        step=10,
        help="シミュレーションの総ステップ数"
    )

    params["mode"] = st.sidebar.selectbox(
        "モード",
        options=["stochastic", "deterministic"],
        index=0,
        help="stochastic: ノイズあり, deterministic (DDIM): ノイズなし"
    )

    params["blur_sigma"] = st.sidebar.slider(
        "ぼかし強度",
        min_value=0.5,
        max_value=5.0,
        value=1.5,
        step=0.1,
        help="描画時のガウシアンぼかしの強さ"
    )

    params["gradient_sigma"] = st.sidebar.slider(
        "勾配計算時のぼかし",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="スコア場計算時のガウシアンぼかし"
    )

    params["save_interval"] = st.sidebar.slider(
        "GIF保存間隔",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help="何ステップごとにGIFフレームを保存するか"
    )

    params["seed"] = st.sidebar.number_input(
        "乱数シード",
        min_value=0,
        value=42,
        step=1,
        help="再現性のための乱数シード"
    )

    return params


def main() -> None:
    """メイン関数"""
    st.set_page_config(
        page_title="Noise Becomes Image",
        layout="wide"
    )

    st.title("Noise Becomes Image")
    st.markdown(
        "ランダムな粒子がスコア場を参照して画像を再構成する過程を可視化"
    )
    st.markdown(
        "**重要**: サンプリング中は元画像を直接参照せず、"
        "事前計算したスコア場のみを使用します"
    )

    # セッション状態を初期化
    initialize_session_state()

    # サイドバーのパラメータ
    params = render_sidebar()

    # 画像アップロード
    uploaded_file = st.file_uploader(
        "画像をアップロード",
        type=["png", "jpg", "jpeg"],
        help="PNG、JPEG形式の画像をアップロードしてください（最大512x512にリサイズされます）"
    )

    if uploaded_file is not None:
        # 画像が変更された場合、再処理
        if st.session_state.last_uploaded_file != uploaded_file:
            st.session_state.last_uploaded_file = uploaded_file

            with st.spinner("画像を読み込んでいます..."):
                # 画像を読み込み
                image = load_and_preprocess_image(uploaded_file)
                st.session_state.original_image = image

                # スコア場を構築
                score_field = build_score_field(
                    image, gradient_sigma=params["gradient_sigma"]
                )
                st.session_state.score_field = score_field

                # 粒子システムとGIFレコーダーをリセット
                st.session_state.particle_system = None
                st.session_state.gif_recorder = None

            st.success(f"画像を読み込みました（サイズ: {image.shape[1]} x {image.shape[0]}）")

        # 2カラムレイアウト
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("元画像")
            st.image(st.session_state.original_image, use_container_width=True)

        with col2:
            st.subheader("再構成中")
            display_placeholder = st.empty()

        # 初期化ボタン
        if st.button("粒子システムを初期化", type="primary"):
            with st.spinner("粒子システムを初期化しています..."):
                # 設定を作成
                config = ParticleSystemConfig(
                    particle_count=params["particle_count"],
                    step_size=params["step_size"],
                    color_step=params["color_step"],
                    color_sharpness=params["color_sharpness"],
                    noise_strength=params["noise_strength"],
                    total_steps=params["total_steps"],
                    mode=params["mode"]
                )

                # 粒子システムを生成
                st.session_state.particle_system = ParticleSystem(
                    config=config,
                    score_field=st.session_state.score_field,
                    seed=params["seed"]
                )

                # GIFレコーダーを生成
                st.session_state.gif_recorder = GifRecorder(max_frames=300)

            st.success("粒子システムを初期化しました")

        # 実行ボタン
        if st.session_state.particle_system is not None:
            if st.button("実行", type="primary"):
                particle_system = st.session_state.particle_system
                gif_recorder = st.session_state.gif_recorder
                score_field = st.session_state.score_field

                # 進捗バー
                progress_bar = st.progress(0)
                status_text = st.empty()

                # シミュレーションループ
                for step in range(params["total_steps"]):
                    # 粒子を更新
                    particle_system.step()

                    # 描画
                    state = particle_system.get_state()
                    rendered = render_particles(
                        positions=state["positions"],
                        colors=state["colors"],
                        width=score_field.width,
                        height=score_field.height,
                        blur_sigma=params["blur_sigma"]
                    )

                    # 表示更新
                    with col2:
                        display_placeholder.image(rendered, use_container_width=True)

                    # 進捗更新
                    progress = (step + 1) / params["total_steps"]
                    progress_bar.progress(progress)
                    status_text.text(f"ステップ {step + 1} / {params['total_steps']}")

                    # GIF記録
                    if step % params["save_interval"] == 0:
                        gif_recorder.add_frame(rendered)

                # 最終フレームを追加
                gif_recorder.add_frame(rendered)

                progress_bar.empty()
                status_text.empty()
                st.success("シミュレーション完了！")

                # GIF保存
                gif_path = "output.gif"
                gif_recorder.save_gif(gif_path, fps=20)

                st.success(f"GIFを保存しました（{gif_recorder.frame_count()}フレーム）")

                # ダウンロードボタン
                with open(gif_path, "rb") as f:
                    st.download_button(
                        label="GIFをダウンロード",
                        data=f,
                        file_name="noise_becomes_image.gif",
                        mime="image/gif"
                    )
    else:
        st.info("画像をアップロードして開始してください")


if __name__ == "__main__":
    main()
