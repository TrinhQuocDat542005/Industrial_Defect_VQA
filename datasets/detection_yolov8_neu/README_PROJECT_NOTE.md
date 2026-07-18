# Nguồn tham khảo NEU-DET và YOLOv8

Thư mục này giữ các thành phần cần thiết từ nguồn `NEU-DET-with-yolov8-main`:

- `data/NEU-DET/`: ảnh và nhãn YOLO cho sáu lỗi bề mặt thép.
- `yolov8n.pt`, `yolov8s.pt`: weight khởi tạo của YOLOv8.
- `LICENSE`, `CITATION.cff`: thông tin giấy phép và trích dẫn gốc.

Mã nguồn chính của bài tập lớn nằm ở project root:

- `notebooks/`: quy trình bốn tuần.
- `scripts/`: đóng gói Kaggle, tạo ảnh output tiếng Việt và kiểm tra project.
- `app.py`: giao diện tiếng Việt.

Không import clone Ultralytics từ thư mục này. Notebook dùng package `ultralytics` cài trong môi trường để tránh trùng lặp mã nguồn và lỗi đường dẫn.
