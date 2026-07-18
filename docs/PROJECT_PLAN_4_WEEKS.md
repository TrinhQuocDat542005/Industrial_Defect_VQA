# Kế hoạch thực hiện dự án trong bốn tuần

## Week01: Khảo sát dữ liệu NEU-DET

**Mục tiêu**

- Kiểm tra cấu trúc ảnh và nhãn YOLO.
- Thống kê sáu lớp lỗi bề mặt thép.
- Chuẩn bị YAML và output EDA.

**Kết quả cần có**

- Thống kê dữ liệu.
- Biểu đồ phân bố lỗi.
- Ảnh mẫu có bounding box và nhãn tiếng Việt.

## Week02: Detection bằng YOLOv8

**Mục tiêu**

- Huấn luyện baseline YOLOv8 trên NEU-DET.
- Đánh giá Precision, Recall, mAP50 và mAP50-95.
- Trực quan hóa dự đoán bằng nhãn tiếng Việt.

**Kết quả cần có**

- Checkpoint `best.pt`.
- Metric detection.
- Ảnh prediction và bảng chi tiết.

## Week03: Segmentation bằng U-Net

**Mục tiêu**

- Phân vùng lỗi mức pixel trên MVTec AD bottle.
- Tối ưu postprocess bằng validation.
- Kiểm soát false positive trên ảnh good.

**Nguyên tắc**

- Threshold và lọc connected component được thực hiện tại model-space 256×256.
- Mask nhị phân đã lọc mới được resize về kích thước ảnh gốc để hiển thị.
- Quality Gate dùng ảnh defect và good độc lập với tập chọn threshold.

**Kết quả cần có**

- U-Net checkpoint.
- `best_postprocess_config.json`.
- Test summary.
- Quality Gate broken_large và good.

## Week04: Multi-modal Captioning + VQA

**Mục tiêu**

- Kết hợp visual embedding, text embedding và đặc trưng từ U-Net predicted mask.
- Tạo caption có cấu trúc.
- Trả lời câu hỏi VQA.
- Đưa ra quyết định quality inspection.

**Luồng xử lý**

```text
Ảnh MVTec bottle
→ U-Net predicted mask
→ mask-derived features
→ TinyViT visual embedding + Transformer text embedding
→ Fusion MLP
→ VQA answer

U-Net predicted mask
→ segmentation safety gate
→ Quality Decision: Chấp nhận hoặc Loại bỏ
```

**Kết quả cần có**

- Fusion checkpoint.
- Feature CSV có `source_of_mask = unet_predicted`.
- Test metric VQA.
- Output Quality Inspection cho ảnh defect và ảnh good.
