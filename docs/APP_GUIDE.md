# Hướng dẫn sử dụng giao diện

## Khởi động

Tại thư mục gốc project, cài môi trường và chạy ứng dụng:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Mở địa chỉ xuất hiện trên terminal, thông thường là:

```text
http://127.0.0.1:7860
```

## Tab 1: Nhận dạng lỗi bề mặt

Tab này dùng YOLOv8 đã huấn luyện trên NEU Surface Defect Database.

Các bước sử dụng:

1. Tải một ảnh bề mặt thép.
2. Chọn ngưỡng tin cậy.
3. Bấm **Nhận dạng lỗi bề mặt**.
4. Xem ảnh có bounding box, nhãn tiếng Việt, độ tin cậy và bảng chi tiết.

| Nhãn NEU-DET | Hiển thị tiếng Việt |
|---|---|
| crazing | Nứt rạn |
| inclusion | Tạp chất lẫn |
| patches | Vết vá |
| pitted_surface | Rỗ bề mặt |
| rolled-in_scale | Vảy cán |
| scratches | Trầy xước |

## Tab 2: Phân vùng lỗi và VQA

Tab này dùng U-Net và Fusion VQA trên ảnh MVTec AD bottle.

Các bước sử dụng:

1. Tải ảnh chai từ MVTec AD bottle.
2. Nhập câu hỏi VQA, mặc định là `Sản phẩm có đạt chất lượng không?`.
3. Bấm **Phân vùng và trả lời**.
4. Xem overlay vùng lỗi, caption có cấu trúc, câu trả lời VQA và quyết định chất lượng.

Quyết định chất lượng dựa trên mask U-Net sau postprocess. Với câu hỏi về chất lượng, câu trả lời hiển thị được đồng bộ với quyết định `Chấp nhận` hoặc `Loại bỏ`.

## Điều kiện cần có

Tab 2 cần các artifact sau:

```text
outputs/week03_segmentation/checkpoints/unet_bottle_best.pt
outputs/week03_segmentation/best_postprocess_config.json
outputs/week04_multimodal/fusion_vqa.pt
```

Khi thiếu artifact, giao diện cần báo rõ thành phần còn thiếu thay vì tạo kết quả suy đoán.
