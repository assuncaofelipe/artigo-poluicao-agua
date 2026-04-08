from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT.parent / "artigo" / "figuras"
DST = ROOT / "figs"


def crop_top(src_name: str, dst_name: str, top: int) -> None:
    im = Image.open(SRC / src_name)
    cropped = im.crop((0, top, im.width, im.height))
    cropped.save(DST / dst_name)


def copy_file(src_name: str, dst_name: str) -> None:
    im = Image.open(SRC / src_name)
    im.save(DST / dst_name)


def main() -> None:
    copy_file("foto-lago-do-serpa.jpeg", "foto_lago_serpa_clean.jpg")
    copy_file("map_basemap_osm_21S_clean.png", "map_basemap_osm_21S_clean_copy.png")
    crop_top("idw_ph.png", "idw_ph_clean_crop.png", 90)
    crop_top("idw_od.png", "idw_od_clean_crop.png", 90)
    crop_top("idw_turb.png", "idw_turb_clean_crop.png", 90)
    print("prepared clean unused figures")


if __name__ == "__main__":
    main()
