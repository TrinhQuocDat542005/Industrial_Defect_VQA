# Đối chiếu dự án với yêu cầu đề tài

| Yêu cầu | Dataset | Mô hình/phương pháp | Module | Đầu ra |
|---|---|---|---|---|
| Phát hiện vết nứt, trầy xước và lỗi bề mặt | NEU Surface Defect | YOLOv8 | Week02 | Bounding box, nhãn tiếng Việt, confidence, metric detection |
| Phân vùng vùng lỗi chi tiết | MVTec AD bottle | U-Net | Week03 | Mask dự đoán, overlay, metric segmentation |
| Visual embedding | MVTec AD bottle | TinyViTEncoder | Week04 | Visual embedding |
| Text embedding | Câu hỏi VQA | TransformerEncoder | Week04 | Text embedding |
| Fusion module | MVTec AD bottle | Fusion MLP | Week04 | Dự đoán đáp án |
| Captioning | MVTec AD bottle | Caption có cấu trúc từ mask | Week04 | Caption tiếng Việt |
| VQA reasoning | MVTec AD bottle | TinyViT + Transformer + Fusion MLP | Week04 | Câu trả lời VQA |
| Quality inspection | MVTec AD bottle | Segmentation safety gate | Week04 | Chấp nhận hoặc Loại bỏ |

## Nguyên tắc phương pháp

- YOLOv8 trên NEU-DET là baseline detection cho ảnh thép.
- U-Net, caption có cấu trúc, VQA và quality inspection dùng MVTec AD bottle trong cùng domain.
- Không trộn ảnh thép NEU-DET với ảnh chai MVTec trong một lần suy luận.
- Trong inference Week04, Fusion sử dụng đặc trưng trích từ **U-Net predicted mask**.
- Ground-truth mask chỉ phục vụ tạo nhãn hoặc đánh giá trong giai đoạn thí nghiệm; không được dùng thay predicted mask khi suy luận.
- Caption có cấu trúc được mô tả đúng bản chất, không được trình bày là generative captioning.
- Quality Decision ưu tiên segmentation safety gate; đáp án VQA thô được lưu riêng để phân biệt với quyết định cuối.
