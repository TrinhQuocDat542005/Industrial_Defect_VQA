from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

NHAN_TIENG_VIET = {
    0: "Ná»©t ráº¡n",
    1: "Táº¡p cháº¥t láº«n",
    2: "Váº¿t vÃ¡",
    3: "Rá»— bá» máº·t",
    4: "Váº£y cÃ¡n",
    5: "Tráº§y xÆ°á»›c",
}
MAU_LOI = {
    0: (235, 64, 52),
    1: (0, 153, 204),
    2: (255, 125, 0),
    3: (255, 193, 7),
    4: (184, 207, 0),
    5: (190, 66, 245),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def find_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "notebooks").is_dir() and (candidate / "datasets").is_dir():
            return candidate
    raise FileNotFoundError("KhÃ´ng tÃ¬m tháº¥y project root Industrial_Defect_VQA.")


def find_font(size: int):
    for candidate in [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def draw_label(draw, box, text, color):
    x1, y1, x2, y2 = box
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    text_font = find_font(18)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=text_font)
    text_height = bottom - top
    text_width = right - left
    label_y = y1 - text_height - 8 if y1 > text_height + 12 else y1 + 3
    draw.rounded_rectangle((x1, label_y, x1 + text_width + 10, label_y + text_height + 8), radius=4, fill=color)
    draw.text((x1 + 5, label_y + 3), text, font=text_font, fill=(10, 10, 10))


def main():
    parser = argparse.ArgumentParser(description="Táº¡o lÆ°á»›i nháº­n dáº¡ng YOLOv8 vá»›i nhÃ£n tiáº¿ng Viá»‡t.")
    parser.add_argument("--so-anh", type=int, default=12, help="Sá»‘ áº£nh test cáº§n hiá»ƒn thá»‹, máº·c Ä‘á»‹nh 12.")
    parser.add_argument("--nguong", type=float, default=0.25, help="NgÆ°á»¡ng confidence.")
    parser.add_argument("--project-root", type=str, default="", help="Project root, máº·c Ä‘á»‹nh tá»± tÃ¬m.")
    args = parser.parse_args()

    start = Path(args.project_root) if args.project_root else Path.cwd()
    root = find_root(start)

    weight = first_existing([
        root / "outputs/week02_detection/yolov8n_neu/weights/best.pt",
        root / "outputs/week02_detection/weights/best.pt",
    ])
    if weight is None:
        raise FileNotFoundError(
            "ChÆ°a cÃ³ best.pt. HÃ£y cháº¡y train Week02 báº±ng Kaggle GPU trÆ°á»›c."
        )

    try:
        from ultralytics import YOLO
    except Exception as error:
        raise RuntimeError("Thiáº¿u ultralytics. Cháº¡y: python -m pip install -r requirements.txt") from error

    test_dir = root / "datasets/detection_yolov8_neu/data/NEU-DET/test/images"
    images = sorted(path for path in test_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"KhÃ´ng tÃ¬m tháº¥y áº£nh test táº¡i: {test_dir}")

    selected = images[: max(1, args.so_anh)]
    model = YOLO(str(weight))
    results = model.predict(
        source=[str(path) for path in selected],
        conf=float(args.nguong),
        device=0 if __import__("torch").cuda.is_available() else "cpu",
        verbose=False,
    )

    tile_width, tile_height = 420, 330
    columns = 4
    rows = int(np.ceil(len(selected) / columns))
    grid = Image.new("RGB", (columns * tile_width, rows * tile_height + 58), (25, 30, 40))
    title_draw = ImageDraw.Draw(grid)
    title_draw.text(
        (18, 15),
        "Káº¾T QUáº¢ NHáº¬N Dáº NG Lá»–I Bá»€ Máº¶T - YOLOv8 (NHÃƒN TIáº¾NG VIá»†T)",
        font=find_font(26),
        fill=(255, 255, 255),
    )

    prediction_rows = []
    for index, (result, image_path) in enumerate(zip(results, selected)):
        original_rgb = result.orig_img[:, :, ::-1]
        tile = Image.fromarray(original_rgb).resize((tile_width, tile_height - 35))
        draw = ImageDraw.Draw(tile)

        scale_x = tile.width / result.orig_img.shape[1]
        scale_y = tile.height / result.orig_img.shape[0]

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                score = float(box.conf.item())
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].cpu().tolist()]
                scaled = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
                label_vi = NHAN_TIENG_VIET.get(class_id, "KhÃ´ng xÃ¡c Ä‘á»‹nh")
                draw_label(draw, scaled, f"{label_vi} {score:.2f}", MAU_LOI.get(class_id, (0, 255, 0)))
                prediction_rows.append(
                    {
                        "anh": image_path.name,
                        "loai_loi_tieng_viet": label_vi,
                        "do_tin_cay": f"{score:.4f}",
                        "x1": f"{x1:.2f}",
                        "y1": f"{y1:.2f}",
                        "x2": f"{x2:.2f}",
                        "y2": f"{y2:.2f}",
                    }
                )

        card = Image.new("RGB", (tile_width, tile_height), (247, 247, 247))
        card.paste(tile, (0, 35))
        card_draw = ImageDraw.Draw(card)
        card_draw.text((8, 8), image_path.name, font=find_font(16), fill=(20, 20, 20))

        column = index % columns
        row = index // columns
        grid.paste(card, (column * tile_width, 58 + row * tile_height))

    output_dir = root / "outputs/week02_detection/predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "du_doan_yolov8_tieng_viet.png"
    csv_path = output_dir / "du_doan_yolov8_tieng_viet.csv"
    grid.save(png_path)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["anh", "loai_loi_tieng_viet", "do_tin_cay", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"ÄÃ£ táº¡o áº£nh lÆ°á»›i: {png_path}")
    print(f"ÄÃ£ táº¡o CSV dá»± Ä‘oÃ¡n: {csv_path}")


if __name__ == "__main__":
    main()

