# Noise Becomes Image

## 概要

ランダムに配置された粒子が、入力画像から事前計算されたスコア場 ∇log p を参照して移動し、元画像の構造と色を再構成する過程を可視化するWebアプリを実装する。

重要な制約：
サンプリング中に元画像のピクセル値を直接参照してはならない。
粒子が参照できるのは事前計算されたスコア場のみとする。

本アプリは拡散モデルの学習済みネットワークを使用しない。
画像から直接スコア関数を計算し、ランジュバンダイナミクスとして粒子を更新する。

目的：
「スコア場のみから構造と色が再構成される」現象の可視化

---

## 入力

* ユーザーが画像をアップロード（PNG/JPEG）
* RGBのまま使用
* 最大512x512へリサイズ

---

## 密度と色のスコア場の構築（前処理のみ画像参照可）

### 輝度密度

Y(x,y) = 0.299R + 0.587G + 0.114B
p(x,y) = (Y + ε) / Σ(Y + ε)

L(x,y) = log(p(x,y))

score_pos = ∇L(x,y)

---

### 色スコア場（各チャンネル独立）

各チャンネルについて確率密度を作る：

p_r(x,y) = (R + ε) / Σ(R + ε)
p_g(x,y) = (G + ε) / Σ(G + ε)
p_b(x,y) = (B + ε) / Σ(B + ε)

L_r = log(p_r)
L_g = log(p_g)
L_b = log(p_b)

score_r = ∇L_r
score_g = ∇L_g
score_b = ∇L_b

これらを保存し、実行中は画像を参照しない。

---

## 粒子の状態

各粒子：

* position: (float x, float y)
* color: (float r, g, b)  初期値はランダム

初期配置：
位置は一様ランダム、色は [0,1] 一様乱数

---

## 更新式

### 位置更新

x_{t+1} = x_t + η * score_pos(x_t) + σ(t) * N(0,1)

### 色更新（スコアに沿って移動）

c_{t+1} = c_t + α * score_color(x_t)

score_color(x_t) = (score_r, score_g, score_b) を位置でサンプリング

色は勾配方向に収束し、直接サンプリングはしない。

ノイズスケジュール：
σ(t) = σ0 * (1 - t/T)

---

## DDIMモード

ノイズ項を除去：

x_{t+1} = x_t + η * score_pos(x_t)

---

## 描画

各フレーム：

1. 全粒子を更新
2. 粒子の現在の色を位置に加算
3. ガウシアンぼかし
4. 正規化して表示

accum_rgb[x,y] += particle.color
count[x,y] += 1
output = accum_rgb / max(count,1)

---

## GIF生成

* 最大300フレーム保存
* 完了後GIF出力

---

## UIパラメータ

* particle_count
* step_size (η)
* color_step (α)
* noise_strength (σ0)
* steps (T)
* mode: stochastic / deterministic
* blur_sigma
* save_interval

---

## 技術制約
* Python
* uv
* Streamlit
* NumPy

---

## Python記述上でのルール
* 型を明記する
* コメントは日本語で記述する
* List, Dict, Tupleではなく、list, dict, tupleを使う

---

## 期待される挙動

初期：色付きノイズ雲
中盤：色と輪郭が結びつく
終盤：元画像に近い配色と構造が出現
