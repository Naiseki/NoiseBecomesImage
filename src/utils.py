"""
ユーティリティ関数
画像の読み込みと前処理を行う
"""

import numpy as np
from PIL import Image
from typing import Any


def load_and_preprocess_image(
    uploaded_file: Any,
    max_size: int = 512
) -> np.ndarray:
    """
    アップロードされた画像を読み込み、前処理を行う

    Args:
        uploaded_file: Streamlitのアップロードファイルオブジェクト
        max_size: 画像の最大サイズ（幅・高さの大きい方）

    Returns:
        (H, W, 3) のRGB画像、値の範囲は[0, 1]
    """
    # PIL.Imageで読み込み
    image = Image.open(uploaded_file)

    # RGBに変換
    if image.mode != "RGB":
        image = image.convert("RGB")

    # アスペクト比を保ちつつリサイズ
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # NumPy配列に変換
    image_array = np.array(image)

    # [0, 1]に正規化
    image_normalized = image_array.astype(np.float32) / 255.0

    return image_normalized
