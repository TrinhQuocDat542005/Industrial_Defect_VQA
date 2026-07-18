import argparse
import zipfile
import shutil
import datetime
from pathlib import Path
import os

def main():
    parser = argparse.ArgumentParser(description="Nhap artifact Kaggle ve local Windows.")
    parser.add_argument("zip_path", help="Duong dan den file Industrial_Defect_VQA_artifacts_kaggle.zip")
    args = parser.parse_args()
    
    zip_path = Path(args.zip_path)
    if not zip_path.exists():
        print(f"[LOI] Khong tim thay file ZIP: {zip_path}")
        return
        
    project_root = Path(__file__).resolve().parent.parent
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    # Backup cu
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for folder_name in ["week03_segmentation", "week04_multimodal"]:
        folder = outputs_dir / folder_name
        if folder.exists():
            backup_path = outputs_dir / f"{folder_name}_backup_{timestamp}"
            shutil.move(str(folder), str(backup_path))
            print(f"Da backup {folder_name} thanh {backup_path.name}")
            
    # Giai nen
    print(f"Tien hanh giai nen tu {zip_path} vao {outputs_dir}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # File ZIP da co chua prefix "outputs/" trong arcname
            # Giai nen vao project_root thi no se tu tao cac subfolder trong outputs
            zip_ref.extractall(project_root)
        print("Giai nen hoan tat!")
    except Exception as e:
        print(f"[LOI] Loi khi giai nen: {e}")
        return
        
    # Kiem tra file bat buoc
    required = [
        "outputs/week03_segmentation/checkpoints/unet_bottle_best.pt",
        "outputs/week03_segmentation/validation_postprocess_search.csv",
        "outputs/week04_multimodal/fusion_vqa.pt",
        "outputs/week04_multimodal/qa_image_features.csv"
    ]
    
    missing = False
    for req in required:
        req_path = project_root / req
        if not req_path.exists():
            print(f"[LOI] Thieu file quan trong: {req}")
            missing = True
            
    if missing:
        print("[THAT BAI] Artifact khong day du. Vui long kiem tra lai qua trinh dong goi tren Kaggle.")
    else:
        print("[THANH CONG] Nhap artifact thanh cong, he thong da san sang!")

if __name__ == "__main__":
    main()
