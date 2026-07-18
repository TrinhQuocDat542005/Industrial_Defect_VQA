# Phát hiện và phân vùng lỗi sản phẩm trong công nghiệp kết hợp Captioning + VQA bằng Multi-modal AI

Bài tập lớn môn **Xử lý ảnh và Thị giác máy tính**.
Dự án cung cấp một quy trình hoàn chỉnh từ nhận dạng lỗi, phân vùng mức pixel, cho đến trả lời câu hỏi trực quan (VQA) và đưa ra quyết định chất lượng (Quality Inspection).

## Tổng quan các module

- **Week02 (Detection):** Xây dựng baseline nhận dạng sáu lỗi bề mặt thép của tập dữ liệu NEU-DET bằng YOLOv8.
- **Week03 (Segmentation):** Phân vùng lỗi mức pixel trên sản phẩm chai lọ (MVTec AD bottle) bằng mô hình U-Net. Threshold và `min_component_area` được tối ưu tự động tìm kiếm (Grid Search) dựa trên Validation `Dice_defect_only`. Toàn bộ quá trình đánh giá và postprocess (lọc nhiễu) được thực hiện nhất quán trên **model space (độ phân giải 256x256)** trước khi resize kết quả về kích thước ảnh gốc.
- **Week04 (Multi-modal VQA):** Xây dựng module VQA đa phương thức. 
  - 100% End-to-End: Dùng **predicted mask** trực tiếp từ U-Net làm dữ liệu đầu vào.
  - Sử dụng TinyViT cho visual embedding và Transformer cho text embedding, hợp nhất bằng Fusion MLP.
  - Hỗ trợ xuất "Caption có cấu trúc", không gọi là generative captioning.
  - Tích hợp Safety Gate: Đảm bảo đồng bộ quyết định chất lượng giữa U-Net segmentation và hệ thống trả lời câu hỏi tự do VQA.

## Nguyên tắc khoa học

- Tuần 2: YOLOv8 + NEU-DET là baseline Detection.
- Tuần 3: U-Net + MVTec `bottle` là Segmentation.
- Tuần 4: chỉ dùng MVTec `bottle`. Dùng mask U-Net end-to-end cho VQA.
- Dự án **không chạy YOLO để học lỗi thép NEU lên ảnh chai MVTec**. Không trộn lẫn 2 tập dữ liệu trong cùng một module suy luận.
- **Lưu ý:** Workspace local chưa có sẵn mô hình đã train. Cần tải artifact mới từ Kaggle.

## Hướng dẫn thao tác cuối (Deploy from Kaggle)

1. **Sau khi huấn luyện xong trên Kaggle**, hãy mở terminal trên Kaggle (hoặc chọn 1 cell) và chạy lệnh đóng gói toàn bộ artifact:
```bash
python scripts/dong_goi_artifact_kaggle.py
```

2. **Tải file artifact về máy local:**
Tải file `outputs/Industrial_Defect_VQA_artifacts_kaggle.zip` từ giao diện Kaggle về máy tính (ví dụ vào thư mục Downloads).

3. **Nhập artifact và khởi động ứng dụng trên máy local (Windows):**
Mở PowerShell tại thư mục dự án và chạy:
```powershell
# Bước 3.1: Nhập và tự động giải nén artifact
python scripts/nhap_artifact_kaggle.py "C:\Users\<YourUser>\Downloads\Industrial_Defect_VQA_artifacts_kaggle.zip"

# Bước 3.2: Kiểm tra tổng quát tính toàn vẹn của dự án
python scripts/kiem_tra_output_cuoi.py

# Bước 3.3: Chạy ứng dụng giao diện web
python app.py
```
Mở đường dẫn `http://127.0.0.1:7860` trên trình duyệt để sử dụng.
