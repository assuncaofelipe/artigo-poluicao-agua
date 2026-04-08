from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figs"


VARIANTS = {
    "map_points_osm_21S.png": {
        "output": "map_points_osm_21S_variant.png",
        "title": "Base Territorial do Lago do Serpa",
        "subtitle": "Pontos georreferenciados de amostragem",
        "dark": "#233d4d",
        "light": "#f4f1de",
        "accent": "#7a9e7e",
    },
    "pred_vs_real_pH_multi.png": {
        "output": "pred_vs_real_pH_multi_variant.png",
        "title": "Leitura Aplicada do pH",
        "subtitle": "Comparação entre observação e estimativa",
        "dark": "#114b5f",
        "light": "#f3fafb",
        "accent": "#4d908e",
    },
    "pred_vs_real_OD_multi.png": {
        "output": "pred_vs_real_OD_multi_variant.png",
        "title": "Leitura Aplicada do Oxigênio Dissolvido",
        "subtitle": "Comparação entre observação e estimativa",
        "dark": "#184e77",
        "light": "#f2f8fc",
        "accent": "#34a0a4",
    },
    "pred_vs_real_TURB_multi.png": {
        "output": "pred_vs_real_TURB_multi_variant.png",
        "title": "Leitura Aplicada da Turbidez",
        "subtitle": "Comparação entre observação e estimativa",
        "dark": "#6b705c",
        "light": "#fcfbf7",
        "accent": "#cb997e",
    },
    "idw_combined.png": {
        "output": "idw_combined_variant.png",
        "title": "Painel Espacial da Qualidade da Água",
        "subtitle": "Variações ambientais interpoladas no Lago do Serpa",
        "dark": "#2f3e46",
        "light": "#f0efeb",
        "accent": "#84a98c",
    },
    "energy_platforms_bar.png": {
        "output": "energy_platforms_bar_variant.png",
        "title": "Comparativo Operacional entre Plataformas",
        "subtitle": "Custo energético por inferência",
        "dark": "#3d405b",
        "light": "#f9f7f3",
        "accent": "#e07a5f",
    },
    "pollution_scatter.png": {
        "output": "pollution_scatter_variant.png",
        "title": "Índice Ambiental nos Pontos de Coleta",
        "subtitle": "Distribuição relativa das condições observadas",
        "dark": "#264653",
        "light": "#f8faf7",
        "accent": "#2a9d8f",
    },
    "pollution_idw.png": {
        "output": "pollution_idw_variant.png",
        "title": "Mancha Espacial do Índice Ambiental",
        "subtitle": "Interpolação da condição relativa no lago",
        "dark": "#5f0f40",
        "light": "#fff7fb",
        "accent": "#9a031e",
    },
}


def hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
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


def stylize_image(image: Image.Image, dark: str, light: str) -> Image.Image:
    rgb = image.convert("RGB")
    gray = ImageOps.grayscale(rgb)
    colorized = ImageOps.colorize(gray, black=hex_rgb(dark), white=hex_rgb(light))
    base = Image.blend(rgb, colorized, 0.58)
    base = ImageEnhance.Contrast(base).enhance(1.08)
    base = ImageEnhance.Sharpness(base).enhance(1.12)
    return base


def make_texture(size: tuple[int, int], accent: str) -> Image.Image:
    width, height = size
    arr = np.random.default_rng(42).integers(0, 14, size=(height, width), dtype=np.uint8)
    texture = Image.fromarray(arr, mode="L")
    texture = ImageEnhance.Contrast(texture).enhance(0.6)
    tex_rgb = ImageOps.colorize(texture, black=(255, 255, 255), white=hex_rgb(accent))
    return tex_rgb


def add_frame(original: Image.Image, title: str, subtitle: str, dark: str, light: str, accent: str) -> Image.Image:
    title_font = load_font(max(28, original.width // 62), bold=True)
    sub_font = load_font(max(18, original.width // 95), bold=False)
    body_font = load_font(max(14, original.width // 120), bold=False)

    pad = max(60, original.width // 28)
    head_h = max(135, original.height // 9)
    foot_h = max(58, original.height // 18)

    canvas = Image.new("RGB", (original.width + pad * 2, original.height + head_h + foot_h + pad), hex_rgb(light))
    texture = make_texture(canvas.size, accent)
    canvas = Image.blend(canvas, texture, 0.08)

    shadow = Image.new("RGBA", (original.width + 24, original.height + 24), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle((0, 0, shadow.width - 1, shadow.height - 1), radius=28, fill=(25, 25, 25, 72))
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    canvas_rgba = canvas.convert("RGBA")
    canvas_rgba.alpha_composite(shadow, (pad - 4, head_h - 2))

    panel = Image.new("RGB", (original.width, original.height), "white")
    panel.paste(original, (0, 0))
    canvas_rgba.alpha_composite(panel.convert("RGBA"), (pad, head_h))

    draw = ImageDraw.Draw(canvas_rgba)
    draw.rounded_rectangle((pad - 10, head_h - 10, pad + original.width + 10, head_h + original.height + 10), radius=24, outline=hex_rgb(accent), width=8)
    draw.rectangle((0, 0, canvas_rgba.width, head_h - 20), fill=hex_rgb(dark))
    draw.text((pad, 28), title, font=title_font, fill=hex_rgb(light))
    draw.text((pad, 78), subtitle, font=sub_font, fill=hex_rgb(light))
    draw.text((pad, head_h + original.height + 18), "Versão gráfica derivada para submissão aplicada", font=body_font, fill=hex_rgb(dark))

    return canvas_rgba.convert("RGB")


def main() -> None:
    for source_name, cfg in VARIANTS.items():
        source = FIGS / source_name
        target = FIGS / cfg["output"]
        image = Image.open(source)
        styled = stylize_image(image, cfg["dark"], cfg["light"])
        final = add_frame(styled, cfg["title"], cfg["subtitle"], cfg["dark"], cfg["light"], cfg["accent"])
        final.save(target, quality=95)
        print(f"created {target.name}")


if __name__ == "__main__":
    main()
