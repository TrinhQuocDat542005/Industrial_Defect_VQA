# Đầu ra theo bốn tuần

Tài liệu này liệt kê các output cần có trong project. Checkpoint và metric phải được sinh từ lần chạy notebook thực tế.

## Week01 - EDA NEU-DET

Thư mục:

```text
outputs/week01_eda/
```

Đầu ra chính:

```text
thong_ke_anh_va_nhan.csv
phan_bo_nhan_theo_lop.csv
phan_bo_loi_tieng_viet.png
mau_bbox_tieng_viet.png
bao_cao_kiem_tra_du_lieu.csv
```

## Week02 - YOLOv8 Detection

Thư mục:

```text
outputs/week02_detection/
```

Đầu ra chính:

```text
weights/best.pt
metrics_yolov8.csv
predictions/du_doan_yolov8_tieng_viet.png
predictions/du_doan_yolov8_tieng_viet.csv
splits/train.txt
splits/val.txt
splits/test.txt
```

Thư mục `yolov8n_neu/` chứa log, biểu đồ và output huấn luyện của Ultralytics.

## Week03 - U-Net Segmentation

Thư mục:

```text
outputs/week03_segmentation/
```

Đầu ra chính:

```text
checkpoints/unet_bottle_best.pt
best_postprocess_config.json
validation_postprocess_search.csv
split_manifest.csv
test_per_image_unet.csv
test_summary_unet.csv
quality_gate_manifest.csv
minh_hoa/quality_gate_broken_large.png
minh_hoa/quality_gate_good.png
```

`best_postprocess_config.json` cần thể hiện threshold và `min_component_area_model_px` được chọn tại model-space 256×256.

## Week04 - Multi-modal Captioning + VQA

Thư mục:

```text
outputs/week04_multimodal/
```

Đầu ra chính:

```text
fusion_vqa.pt
qa_image_features.csv
qa_image_split_manifest.csv
test_metrics_vqa.csv
test_predictions_vqa.csv
ket_qua_kiem_tra/w4_broken_large_ket_qua_kiem_tra_tieng_viet.json
ket_qua_kiem_tra/w4_broken_large_ket_qua_kiem_tra_tieng_viet.png
ket_qua_kiem_tra/w4_good_ket_qua_kiem_tra_tieng_viet.json
ket_qua_kiem_tra/w4_good_ket_qua_kiem_tra_tieng_viet.png
```

## Kiểm tra trước khi nộp

- Week02 có checkpoint và metric detection.
- Week03 có checkpoint U-Net, config postprocess và kết quả Quality Gate trên ảnh defect/good.
- Week04 dùng `source_of_mask = unet_predicted` trong `qa_image_features.csv`.
- Với câu hỏi về chất lượng, kiểm tra cả hai file JSON:
  - `broken_large` cần có `quality_decision = "Loại bỏ"`.
  - `good` cần có `quality_decision = "Chấp nhận"`.
- Báo cáo accuracy VQA theo đúng `test_metrics_vqa.csv`; không thay thế bằng kết quả Quality Inspection.
