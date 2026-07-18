# Chạy project qua VSCode kết nối Kaggle GPU

## 1. Nguyên tắc

Khi chọn kernel `kaggle.net` trong VSCode, code chạy trên Kaggle. Thư mục project trên Windows không tự đồng bộ với môi trường Kaggle.

Quy trình:

```text
Project local Windows
→ source ZIP
→ Kaggle Input
→ /kaggle/working
→ chạy notebook bằng GPU
→ artifact ZIP
→ import artifact về project local
```

## 2. Chuẩn bị source ZIP trên Windows

Tạo ZIP source theo định dạng POSIX để Kaggle chấp nhận đường dẫn. Không dùng `Compress-Archive` nếu ZIP tạo ra dùng dấu `\`.

Từ PowerShell tại `D:\Project\IP_CV`, tạo ZIP bằng Python:

```powershell
@'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

project_root = Path(r"D:\Project\IP_CV\Industrial_Defect_VQA")
zip_path = Path(r"D:\Project\IP_CV\Industrial_Defect_VQA_Kaggle.zip")

skip_dirs = {".git", "__pycache__", ".ipynb_checkpoints"}

with ZipFile(zip_path, "w", ZIP_DEFLATED) as archive:
    for path in project_root.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(project_root)
        if any(part in skip_dirs for part in relative_path.parts):
            continue
        if path.suffix.lower() in {".pyc", ".zip"}:
            continue
        archive.write(path, relative_path.as_posix())

print(zip_path)
'@ | python -
```

Upload `Industrial_Defect_VQA_Kaggle.zip` lên Kaggle dưới dạng Dataset. Mỗi lần thay đổi notebook, script, `app.py` hoặc `requirements.txt`, tạo ZIP mới và upload bằng **New Version**.

## 3. Chuẩn bị Kaggle

Trong Kaggle Notebook:

1. Bật `Settings → Accelerator → GPU`.
2. Add Input dataset source ZIP.
3. Add Input dataset MVTec AD có category `bottle`.
4. Giải nén source ZIP vào `/kaggle/working/Industrial_Defect_VQA`.

Cell khởi tạo tối thiểu:

```python
from pathlib import Path
from zipfile import ZipFile
import subprocess
import sys

input_root = Path("/kaggle/input")
project_zip = next(
    path for path in input_root.rglob("*.zip")
    if "industrial_defect_vqa" in path.name.lower()
)

project_root = Path("/kaggle/working/Industrial_Defect_VQA")
project_root.mkdir(parents=True, exist_ok=True)

with ZipFile(project_zip) as archive:
    archive.extractall(project_root)

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "-r",
    str(project_root / "requirements.txt"),
])

print("PROJECT_ROOT =", project_root)
```

Kiểm tra GPU:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## 4. Chọn kernel trong VSCode

Trong mỗi notebook:

```text
Select Kernel → chọn kernel có kaggle.net
```

Không chọn kernel local khi cần chạy `/kaggle/working` hoặc GPU Kaggle.

## 5. Thứ tự chạy notebook

Chạy theo thứ tự:

```text
Week01_Setup_EDA_NEU.ipynb
→ Week02_YOLOv8_Detection.ipynb
→ Week03_UNet_MVTec_Segmentation.ipynb
→ Week04_Multimodal_Captioning_VQA.ipynb
```

Nếu chỉ sửa segmentation hoặc VQA, chạy lại Week03 rồi Week04.

Trong mỗi notebook:

```text
Restart Kernel → Run All
```

## 6. Điều kiện chuyển từ Week03 sang Week04

Week03 cần kiểm tra đồng thời:

```text
broken_large/000.png → Phát hiện lỗi → Loại bỏ
good/000.png → Không báo lỗi giả → Chấp nhận
```

Chỉ khi hai Quality Gate đạt, mới tiếp tục Week04.

## 7. Đóng gói artifact

Sau khi Week03 và Week04 chạy xong, tại Kaggle:

```python
import subprocess
import sys

subprocess.check_call([
    sys.executable,
    "/kaggle/working/Industrial_Defect_VQA/scripts/dong_goi_artifact_kaggle.py",
    "--project-root",
    "/kaggle/working/Industrial_Defect_VQA",
    "--runtime-root",
    "/kaggle/working/Industrial_Defect_VQA_runtime",
])
```

Artifact được tạo tại:

```text
/kaggle/working/Industrial_Defect_VQA/outputs/Industrial_Defect_VQA_artifacts_kaggle.zip
```

Tải artifact về Windows.

## 8. Nhập artifact về project local

Mở PowerShell tại thư mục project:

```powershell
cd D:\Project\IP_CV\Industrial_Defect_VQA

python .\scripts
hap_artifact_kaggle.py `
  "$env:USERPROFILE\Downloads\Industrial_Defect_VQA_artifacts_kaggle.zip"

python .\scripts\kiem_tra_output_cuoi.py
```

Sau khi kiểm tra output và quality-gate artifacts, chạy giao diện:

```powershell
python app.py
```
