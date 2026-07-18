from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

KAGGLE_DATASET = "ipythonx/mvtec-ad"


def find_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "notebooks").is_dir() and (candidate / "datasets").is_dir():
            return candidate
    raise FileNotFoundError("KhÃ´ng tÃ¬m tháº¥y project root Industrial_Defect_VQA.")


def valid_bottle(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "train" / "good").is_dir()
        and (path / "test").is_dir()
        and (path / "ground_truth").is_dir()
    )


def find_bottle(source: Path) -> Path | None:
    direct = source / "bottle"
    if valid_bottle(direct):
        return direct

    for candidate in source.glob("**/bottle"):
        if valid_bottle(candidate):
            return candidate
    return None


def copy_bottle(source: Path, destination: Path) -> None:
    if destination.exists():
        if valid_bottle(destination):
            print(f"MVTec bottle Ä‘Ã£ cÃ³ sáºµn: {destination}")
            return
        raise RuntimeError(f"Folder Ä‘Ã­ch Ä‘Ã£ tá»“n táº¡i nhÆ°ng cáº¥u trÃºc khÃ´ng há»£p lá»‡: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    print(f"ÄÃ£ Ä‘áº·t MVTec bottle táº¡i: {destination}")


def main():
    parser = argparse.ArgumentParser(
        description="Chuáº©n bá»‹ MVTec AD category bottle cho Week03 vÃ  Week04."
    )
    parser.add_argument("--project-root", default="", help="Project root, máº·c Ä‘á»‹nh tá»± tÃ¬m.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Táº£i MVTec AD tá»« Kaggle API. Cáº§n kaggle.json/Kaggle API token vÃ  Internet.",
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="XÃ¡c nháº­n dÃ¹ng MVTec AD cho há»c táº­p phi thÆ°Æ¡ng máº¡i theo license cá»§a dataset.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Giá»¯ file ZIP vÃ  thÆ° má»¥c cache sau khi hoÃ n thÃ nh.",
    )
    args = parser.parse_args()

    root = find_root(Path(args.project_root) if args.project_root else Path.cwd())
    destination = root / "datasets/mvtec_ad/bottle"
    if valid_bottle(destination):
        print(f"MVTec bottle Ä‘Ã£ sáºµn sÃ ng: {destination}")
        return

    candidate_sources = [
        Path("/kaggle/input/mvtec-ad"),
        Path("/kaggle/input/mvtec-anomaly-detection"),
        Path("/kaggle/input"),
        root / "data/mvtec_ad",
    ]
    for candidate in candidate_sources:
        if candidate.exists():
            found = find_bottle(candidate)
            if found:
                copy_bottle(found, destination)
                return

    if not args.download:
        raise FileNotFoundError(
            "ChÆ°a tÃ¬m tháº¥y MVTec bottle. TrÃªn Kaggle hÃ£y Add Data dataset 'ipythonx/mvtec-ad', "
            "hoáº·c cháº¡y láº¡i vá»›i --download --accept-license náº¿u Ä‘Ã£ cáº¥u hÃ¬nh Kaggle API."
        )

    if not args.accept_license:
        raise PermissionError(
            "MVTec AD dÃ¹ng cho há»c táº­p phi thÆ°Æ¡ng máº¡i. Cháº¡y láº¡i vá»›i --accept-license Ä‘á»ƒ xÃ¡c nháº­n."
        )

    cache = root / ".cache_mvtec"
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / "mvtec-ad.zip"

    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(cache), "--force"],
            check=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "KhÃ´ng cÃ³ lá»‡nh kaggle. CÃ i báº±ng: python -m pip install kaggle"
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Kaggle API khÃ´ng táº£i Ä‘Æ°á»£c dataset. Kiá»ƒm tra kaggle.json/API token, Internet vÃ  quyá»n truy cáº­p dataset."
        ) from error

    archives = list(cache.glob("*.zip"))
    if not archives:
        raise FileNotFoundError("Kaggle CLI khÃ´ng táº¡o file ZIP MVTec AD trong cache.")
    archive_path = max(archives, key=lambda path: path.stat().st_mtime)

    extracted = cache / "extracted"
    if not extracted.exists():
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extracted)

    found = find_bottle(extracted)
    if not found:
        raise FileNotFoundError(
            "ÄÃ£ táº£i nhÆ°ng khÃ´ng tÃ¬m tháº¥y category bottle Ä‘Ãºng cáº¥u trÃºc. Kiá»ƒm tra dataset Kaggle."
        )

    copy_bottle(found, destination)
    if not args.keep_cache:
        shutil.rmtree(cache, ignore_errors=True)
        print("ÄÃ£ xÃ³a cache táº£i vá» Ä‘á»ƒ tiáº¿t kiá»‡m dung lÆ°á»£ng.")


if __name__ == "__main__":
    main()

