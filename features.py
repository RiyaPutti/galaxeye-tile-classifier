import numpy as np
from PIL import Image

N_HIST_BINS = 16

def extract_features(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((64, 64))
    arr = np.asarray(img).astype(np.float32) / 255.0  # (64, 64, 3)

    # 1. Per-channel color histograms
    hist_features = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=N_HIST_BINS, range=(0, 1))
        hist_features.append(hist.astype(np.float32) / hist.sum())
    hist_features = np.concatenate(hist_features)

    # 2. Texture proxy: gradient magnitude stats
    gray = arr.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    texture_features = np.array(
        [gx.mean(), gx.std(), gy.mean(), gy.std()], dtype=np.float32
    )

    # 3. Global color stats
    color_stats = np.concatenate(
        [arr.mean(axis=(0, 1)), arr.std(axis=(0, 1))]
    ).astype(np.float32)

    return np.concatenate([hist_features, texture_features, color_stats])