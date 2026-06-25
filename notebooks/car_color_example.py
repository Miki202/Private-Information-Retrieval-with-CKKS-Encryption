from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

CENTER_CROP_RATIO = 0.60     
RESIZE_TO         = 128      
ACHROMATIC_SAT    = 0.15     

def _rgb_to_hsv_arr(rgb01: np.ndarray) -> np.ndarray:
    """Vectorised RGB -> HSV (all in [0, 1]). Input shape (N, 3)."""
    r, g, b = rgb01[:, 0], rgb01[:, 1], rgb01[:, 2]
    maxc = np.max(rgb01, axis=1)
    minc = np.min(rgb01, axis=1)
    v = maxc

    delta = maxc - minc
    s = np.where(maxc > 0, delta / np.maximum(maxc, 1e-9), 0.0)
    h = np.zeros_like(maxc)
    mask = delta > 1e-9

    rc = np.where(mask, (maxc - r) / np.maximum(delta, 1e-9), 0.0)
    gc = np.where(mask, (maxc - g) / np.maximum(delta, 1e-9), 0.0)
    bc = np.where(mask, (maxc - b) / np.maximum(delta, 1e-9), 0.0)

    h = np.where(r == maxc, bc - gc, h)
    h = np.where(g == maxc, 2.0 + rc - bc, h)
    h = np.where(b == maxc, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    h = np.where(mask, h, 0.0)

    return np.stack([h, s, v], axis=1)


def _classify_hsv(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[:, 0] * 360.0, hsv[:, 1], hsv[:, 2]

    labels = np.empty(len(hsv), dtype=object)

    achromatic = s < ACHROMATIC_SAT
    labels[achromatic & (v < 0.15)]               = "black"
    labels[achromatic & (v >= 0.15) & (v < 0.40)] = "dark_gray"
    labels[achromatic & (v >= 0.40) & (v < 0.70)] = "gray"
    labels[achromatic & (v >= 0.70) & (v < 0.90)] = "silver"
    labels[achromatic & (v >= 0.90)]              = "white"

    chrom = ~achromatic
    for lo, hi, name in [
        (  0,  15, "red"),
        ( 15,  40, "orange"),
        ( 40,  65, "yellow"),
        ( 65, 170, "green"),
        (170, 200, "cyan"),
        (200, 250, "blue"),
        (250, 290, "purple"),
        (290, 345, "magenta"),
        (345, 360, "red"),
    ]:
        labels[chrom & (h >= lo) & (h < hi)] = name

    return labels


def extract_color(
    pil_car: Image.Image,
    center_crop_ratio: float = CENTER_CROP_RATIO,
) -> dict:
    pil_car = pil_car.convert("RGB")

    w, h = pil_car.size
    cw, ch = int(w * center_crop_ratio), int(h * center_crop_ratio)
    x1, y1 = (w - cw) // 2, (h - ch) // 2
    crop = pil_car.crop((x1, y1, x1 + cw, y1 + ch))
    crop.thumbnail((RESIZE_TO, RESIZE_TO), Image.BILINEAR)
    rgb = np.asarray(crop, dtype=np.float32).reshape(-1, 3) / 255.0
    hsv = _rgb_to_hsv_arr(rgb)
    labels = _classify_hsv(hsv)
    counts = Counter(labels.tolist())
    total = sum(counts.values())
    distribution = {k: v / total for k, v in counts.most_common()}
    top_name, top_count = counts.most_common(1)[0]
    winners = rgb[labels == top_name]
    median_rgb = (np.median(winners, axis=0) * 255).astype(int)
    hex_color = "#{:02X}{:02X}{:02X}".format(*median_rgb)

    return {
        "name":         top_name,
        "share":        top_count / total,
        "distribution": distribution,
        "rgb_estimate": tuple(int(c) for c in median_rgb),
        "hex":          hex_color,
    }

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python car_color_example.py <cropped_car1.jpg> [<cropped_car2.jpg> ...]")
        sys.exit(0)

    for path in sys.argv[1:]:
        img = Image.open(path)
        result = extract_color(img)
        print(f"\n=== {Path(path).name} ===")
        print(f"  colour:       {result['name']}   ({result['share']*100:.1f}% of pixels)")
        print(f"  RGB estimate: {result['rgb_estimate']}   hex={result['hex']}")
        print( "  distribution: " + ", ".join(
            f"{k}={v*100:.0f}%" for k, v in list(result['distribution'].items())[:5]
        ))
