from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageColor, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent / "mendeley_dataset" / "original_dataset" / "lago_serpa_raw.csv"
BASEMAP = ROOT.parent / "artigo" / "figuras" / "map_basemap_osm_21S_clean.png"
OUT = ROOT / "figs" / "idw_triptych_redesigned.png"
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS

PANELS = [
    ("pH", "pH", "pH", ["#2b3f8d", "#2f7faa", "#49b77e", "#dbe45f"]),
    ("OD", "Oxigênio dissolvido", "mg/L", ["#103e52", "#1f8a8a", "#7bd389", "#f0cf65"]),
    ("TURB", "Turbidez", "NTU", ["#40306b", "#6f5aa8", "#c06c84", "#f2a65a"]),
]


def load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def hex_to_rgb(value: str) -> np.ndarray:
    return np.array(ImageColor.getrgb(value), dtype=np.float32)


def make_palette(colors: list[str], steps: int = 256) -> np.ndarray:
    anchors = np.linspace(0.0, 1.0, len(colors))
    xs = np.linspace(0.0, 1.0, steps)
    rgb = np.stack([hex_to_rgb(c) for c in colors], axis=0)
    out = np.zeros((steps, 3), dtype=np.uint8)
    for i in range(3):
        out[:, i] = np.interp(xs, anchors, rgb[:, i]).astype(np.uint8)
    return out


def idw_grid(x: np.ndarray, y: np.ndarray, z: np.ndarray, width: int, height: int, extent: tuple[float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax = extent
    gx = np.linspace(xmin, xmax, width)
    gy = np.linspace(ymax, ymin, height)
    xx, yy = np.meshgrid(gx, gy)

    dx = xx[..., None] - x[None, None, :]
    dy = yy[..., None] - y[None, None, :]
    dist2 = dx * dx + dy * dy
    dist2 = np.maximum(dist2, 1e-9)
    weights = 1.0 / dist2
    return np.sum(weights * z[None, None, :], axis=2) / np.sum(weights, axis=2)


def normalize(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return np.clip((values - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)


def draw_colorbar(draw: ImageDraw.ImageDraw, palette: np.ndarray, box: tuple[int, int, int, int], ticks: list[float], vmin: float, vmax: float) -> None:
    x0, y0, x1, y1 = box
    height = y1 - y0
    for row in range(height):
        t = 1.0 - row / max(height - 1, 1)
        idx = int(t * (len(palette) - 1))
        color = tuple(int(v) for v in palette[idx])
        draw.line((x0, y0 + row, x1, y0 + row), fill=color, width=1)

    draw.rectangle(box, outline="#6d6d6d", width=1)
    tick_font = load_font(18, bold=False)
    for tick in ticks:
        frac = normalize(np.array([tick]), vmin, vmax)[0]
        y = y1 - int(frac * height)
        draw.line((x1 + 2, y, x1 + 10, y), fill="#444444", width=1)
        draw.text((x1 + 16, y), f"{tick:.2f}".rstrip("0").rstrip("."), fill="#3a3a3a", font=tick_font, anchor="lm")


def main() -> None:
    df = pd.read_csv(DATA)
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    pad_x = (xmax - xmin) * 0.06
    pad_y = (ymax - ymin) * 0.06
    extent = (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)

    basemap = Image.open(BASEMAP).convert("RGB")
    basemap = basemap.resize((620, 620), RESAMPLE)
    gray = basemap.convert("L")
    gray_rgb = Image.merge("RGB", (gray, gray, gray))
    base_arr = np.asarray(gray_rgb).astype(np.float32)
    base_arr = 255.0 - (255.0 - base_arr) * 0.55
    base_arr = np.clip(base_arr, 0, 255).astype(np.uint8)

    note_font = load_font(20, bold=False)
    axis_font = load_font(22, bold=False)

    panel_w = 760
    panel_h = 820
    map_x0, map_y0 = 34, 34
    map_w, map_h = 620, 620
    colorbar_x0 = 672
    left_margin = 72
    bottom_margin = 54
    canvas = Image.new("RGB", (panel_w * 3 + 70 + left_margin, panel_h + bottom_margin), "#ffffff")
    draw = ImageDraw.Draw(canvas)

    for idx, (column, title, unit, colors) in enumerate(PANELS):
        palette = make_palette(colors)
        values = df[column].to_numpy(dtype=float)
        grid = idw_grid(x, y, values, map_w, map_h, extent)
        vmin, vmax = values.min(), values.max()
        scaled = normalize(grid, vmin, vmax)
        palette_idx = np.clip((scaled * 255).astype(np.int32), 0, 255)
        heatmap = palette[palette_idx]

        overlay_alpha = 0.56
        merged = (base_arr * (1.0 - overlay_alpha) + heatmap * overlay_alpha).astype(np.uint8)
        panel_map = Image.fromarray(merged, mode="RGB")

        panel = Image.new("RGB", (panel_w, panel_h), "#ffffff")
        panel_draw = ImageDraw.Draw(panel)
        panel.paste(panel_map, (map_x0, map_y0))
        panel_draw.rectangle((map_x0, map_y0, map_x0 + map_w, map_y0 + map_h), outline="#8a8a8a", width=1)

        px = map_x0 + ((x - extent[0]) / (extent[1] - extent[0]) * map_w)
        py = map_y0 + ((extent[3] - y) / (extent[3] - extent[2]) * map_h)
        for sx, sy in zip(px, py):
            panel_draw.ellipse((sx - 4, sy - 4, sx + 4, sy + 4), fill="#1b1b1b", outline="#f5f5f5", width=1)

        ticks = np.linspace(vmin, vmax, 5).tolist()
        draw_colorbar(panel_draw, palette, (colorbar_x0, map_y0 + 16, colorbar_x0 + 22, map_y0 + map_h - 16), ticks, vmin, vmax)
        panel_draw.text((map_x0, map_y0 + map_h + 24), f"{title} ({unit})", fill="#4a4a4a", font=note_font)

        x_off = left_margin + idx * panel_w + 20 + idx * 5
        canvas.paste(panel, (x_off, 0))

    x_axis_text = "x: coordenada leste (UTM 21S, m)"
    y_axis_text = "y: coordenada norte (UTM 21S, m)"
    draw.text((canvas.width // 2, panel_h + 18), x_axis_text, fill="#4a4a4a", font=axis_font, anchor="ma")

    y_img = Image.new("RGBA", (420, 60), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_img)
    y_draw.text((210, 30), y_axis_text, fill="#4a4a4a", font=axis_font, anchor="mm")
    y_img = y_img.rotate(90, expand=True)
    canvas.paste(y_img, (10, (panel_h - y_img.height) // 2), y_img)

    canvas.save(OUT, quality=95)
    print(f"created {OUT.name}")


if __name__ == "__main__":
    main()
