from __future__ import annotations
import argparse
import zipfile
from pathlib import Path

def find_source_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "notebooks").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise FileNotFoundError("Khong tim thay source root Industrial_Defect_VQA.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Dong goi checkpoint va output tu Kaggle de chay app local.")
    parser.add_argument("--project-root", default="")
    parser.add_argument(
        "--runtime-root",
        default="/kaggle/working/Industrial_Defect_VQA_runtime",
        help="Noi cac notebook ghi outputs khi chay Kaggle.",
    )
    args = parser.parse_args()

    source_root = find_source_root(Path(args.project_root) if args.project_root else Path.cwd())
    runtime_root = Path(args.runtime_root)
    output_source = runtime_root / "outputs" if (runtime_root / "outputs").exists() else source_root / "outputs"
    output_zip = source_root / "outputs" / "Industrial_Defect_VQA_artifacts_kaggle.zip"
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    includes = [
        output_source / "week02_detection",
        output_source / "week03_segmentation",
        output_source / "week04_multimodal",
    ]

    count = 0
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in includes:
            if not folder.exists():
                continue
            for file in folder.rglob("*"):
                if file.is_file():
                    arcname = (Path("outputs") / file.relative_to(output_source)).as_posix()
                    archive.write(file, arcname)
                    count += 1

    print(f"Da tao thanh cong file ZIP: {output_zip}")
    print(f"Tong so file output duoc dong goi: {count}")

if __name__ == "__main__":
    main()
