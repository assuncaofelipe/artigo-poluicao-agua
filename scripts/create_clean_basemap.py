#!/usr/bin/env python3
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt, label
import csv


def crop_white_margins(img: np.ndarray, thr: float = 0.98) -> np.ndarray:
    rgb = img[..., :3]
    non_white = (rgb < thr).any(axis=2)
    if not non_white.any():
        return img
    ys, xs = np.where(non_white)
    return img[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def marker_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # Blue marker candidates.
    base = (b > 0.50) & ((b - r) > 0.08) & ((b - g) > -0.05)

    comp, n = label(base)
    keep = np.zeros_like(base, dtype=bool)
    for i in range(1, n + 1):
        ys, xs = np.where(comp == i)
        area = len(xs)
        if area < 30 or area > 1600:
            continue
        w = xs.max() - xs.min() + 1
        h = ys.max() - ys.min() + 1
        if w < 6 or h < 6:
            continue
        ratio = w / h
        # Keep compact blobs (markers), drop elongated map textures.
        if 0.65 <= ratio <= 1.6:
            keep[ys, xs] = True

    keep = binary_dilation(keep, iterations=2)
    return keep


def nearest_fill(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # For each masked pixel, copy value from nearest unmasked neighbor.
    _, idx = distance_transform_edt(mask, return_indices=True)
    out = img.copy()
    out[mask] = img[idx[0][mask], idx[1][mask]]
    return out


def build_point_mask_from_data(
    h: int, w: int, xs: np.ndarray, ys: np.ndarray, radius_px: int, flip_y: bool
) -> np.ndarray:
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    px = (xs - xmin) / (xmax - xmin + 1e-12) * (w - 1)
    py_norm = (ys - ymin) / (ymax - ymin + 1e-12)
    py = py_norm * (h - 1)
    if not flip_y:
        py = (h - 1) - py

    yy, xx = np.ogrid[:h, :w]
    mask = np.zeros((h, w), dtype=bool)
    r2 = radius_px * radius_px
    for cx, cy in zip(px, py):
        mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= r2
    return mask


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    src = repo / "figuras" / "map_points_osm_21S.png"
    dst1 = repo / "figuras" / "map_basemap_osm_21S_clean.png"
    dst2 = repo / "artigo" / "figuras" / "map_basemap_osm_21S_clean.png"

    img = np.asarray(Image.open(src).convert("RGB"), dtype=np.float32) / 255.0
    img = crop_white_margins(img)
    h, w = img.shape[:2]

    # Load sampling coordinates used to draw point markers.
    data_path = repo / "scripts" / "data" / "dados_serpa.csv"
    x_list, y_list = [], []
    with data_path.open(encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            x_list.append(float(row["x"].replace(",", ".")))
            y_list.append(float(row["y"].replace(",", ".")))
    xs = np.asarray(x_list, dtype=np.float32)
    ys = np.asarray(y_list, dtype=np.float32)

    # Determine y-axis orientation that best aligns with blue markers.
    m1 = build_point_mask_from_data(h, w, xs, ys, radius_px=11, flip_y=False)
    m2 = build_point_mask_from_data(h, w, xs, ys, radius_px=11, flip_y=True)
    b = img[..., 2]
    score1 = float(b[m1].mean()) if m1.any() else 0.0
    score2 = float(b[m2].mean()) if m2.any() else 0.0
    mask = m1 if score1 >= score2 else m2

    # Union with color-based mask to catch slight misalignment.
    mask |= marker_mask(img)
    mask = binary_dilation(mask, iterations=1)
    clean = nearest_fill(img, mask)

    arr = (np.clip(clean, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(arr).save(dst1)
    Image.fromarray(arr).save(dst2)

    print(f"saved {dst1}")
    print(f"saved {dst2}")
    print(f"masked ratio: {mask.mean():.4f}")


if __name__ == "__main__":
    main()
