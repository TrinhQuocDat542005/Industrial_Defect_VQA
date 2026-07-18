# Tài liệu dự án

Thư mục này tập hợp tài liệu vận hành và đối chiếu yêu cầu cho dự án **Industrial_Defect_VQA**.

## Danh mục tài liệu

- `APP_GUIDE.md`: hướng dẫn sử dụng giao diện Gradio.
- `DATASET_SETUP.md`: vị trí và vai trò của NEU-DET và MVTec AD bottle.
- `HUONG_DAN_CHAY_TU_VSCODE_KAGGLE.md`: quy trình chạy notebook bằng Kaggle GPU và nhập artifact về máy local.
- `OUTPUTS_4_WEEKS.md`: đầu ra cần kiểm tra theo từng tuần.
- `PROJECT_PLAN_4_WEEKS.md`: phạm vi công việc của bốn tuần.
- `TOPIC_ALIGNMENT.md`: đối chiếu các module với yêu cầu đề tài.

## Nguyên tắc sử dụng

- Week02 dùng **NEU-DET + YOLOv8** cho bài toán detection lỗi bề mặt thép.
- Week03 và Week04 dùng **MVTec AD bottle + U-Net** cho segmentation, caption có cấu trúc, VQA và quality inspection.
- Không dùng YOLO đã học trên NEU-DET để suy luận trực tiếp trên ảnh MVTec bottle.
- Checkpoint, metric và output thực nghiệm được lưu trong `outputs/`.
