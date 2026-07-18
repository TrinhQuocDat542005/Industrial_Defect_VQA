from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 256
MAX_QUESTION_LENGTH = 16

NHAN_TIENG_VIET = {
    0: "Nứt rạn",
    1: "Tạp chất lẫn",
    2: "Vết vá",
    3: "Rỗ bề mặt",
    4: "Vảy cán",
    5: "Trầy xước",
}

MAU_LOI = {
    0: (235, 64, 52),
    1: (0, 153, 204),
    2: (255, 125, 0),
    3: (255, 193, 7),
    4: (184, 207, 0),
    5: (190, 66, 245),
}

_YOLO_MODEL = None
_UNET_MODEL = None
_FUSION_MODEL = None
_POSTPROCESS_CONFIG = None


# ==========================================================
# 1. PATHS, FONT VÀ CẤU HÌNH
# ==========================================================

def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]

    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)

    return ImageFont.load_default()


def yolo_weight_path() -> Path | None:
    return first_existing([
        PROJECT_ROOT / "outputs" / "week02_detection" / "weights" / "best.pt",
    ])


def unet_weight_path() -> Path | None:
    return first_existing([
        PROJECT_ROOT
        / "outputs"
        / "week03_segmentation"
        / "checkpoints"
        / "unet_bottle_best.pt",
    ])


def fusion_weight_path() -> Path | None:
    return first_existing([
        PROJECT_ROOT / "outputs" / "week04_multimodal" / "fusion_vqa.pt",
    ])


def load_postprocess_config() -> dict[str, Any]:
    """
    Đọc config Week03 thật.
    Không fallback threshold 0.5 hoặc min_area 0 để tránh quay lại logic cũ.
    """
    global _POSTPROCESS_CONFIG

    if _POSTPROCESS_CONFIG is not None:
        return _POSTPROCESS_CONFIG

    config_path = (
        PROJECT_ROOT
        / "outputs"
        / "week03_segmentation"
        / "best_postprocess_config.json"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            "Thiếu outputs/week03_segmentation/best_postprocess_config.json. "
            "Hãy import artifact Kaggle mới nhất."
        )

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError(
            f"Không đọc được best_postprocess_config.json: {error}"
        ) from error

    required_keys = {
        "threshold",
        "min_component_area_model_px",
        "postprocess_space",
    }

    missing_keys = sorted(required_keys - set(config.keys()))

    if missing_keys:
        raise RuntimeError(
            "Config Week03 thiếu khóa bắt buộc: "
            + ", ".join(missing_keys)
        )

    if str(config["postprocess_space"]).strip().lower() != "model_resolution":
        raise RuntimeError(
            "Config Week03 không dùng model_resolution. "
            "App yêu cầu postprocess trong model-space trước khi resize."
        )

    if "selection_status" not in config:
        raise RuntimeError(
            "Config Week03 thiếu selection_status. "
            "Hãy import artifact Week03 mới có config đã được validation."
        )

    selection_status = str(
        config["selection_status"]
    ).strip().lower()

    if selection_status != "valid":
        raise RuntimeError(
            "Config Week03 chưa được đánh dấu valid: "
            f"{selection_status!r}"
        )

    try:
        threshold = float(config["threshold"])
        min_area_model_px = int(config["min_component_area_model_px"])
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "threshold hoặc min_component_area_model_px không hợp lệ."
        ) from error

    if not 0.0 <= threshold <= 1.0:
        raise RuntimeError(
            f"threshold={threshold} không nằm trong khoảng [0, 1]."
        )

    if min_area_model_px <= 0:
        raise RuntimeError(
            "min_component_area_model_px phải lớn hơn 0. "
            "Không dùng cấu hình cũ min_area = 0."
        )

    reference_size = config.get(
        "reference_image_size",
        [IMAGE_SIZE, IMAGE_SIZE],
    )

    if (
        not isinstance(reference_size, (list, tuple))
        or len(reference_size) != 2
        or int(reference_size[0]) != IMAGE_SIZE
        or int(reference_size[1]) != IMAGE_SIZE
    ):
        raise RuntimeError(
            "Config Week03 không khớp IMAGE_SIZE=256."
        )

    _POSTPROCESS_CONFIG = {
        **config,
        "threshold": threshold,
        "min_component_area_model_px": min_area_model_px,
        "reference_image_size": [IMAGE_SIZE, IMAGE_SIZE],
    }

    return _POSTPROCESS_CONFIG


# ==========================================================
# 2. MODEL DEFINITIONS
# ==========================================================

class DoubleConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class UNetSmall(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_1 = DoubleConv(3, 32)
        self.pool_1 = nn.MaxPool2d(2)

        self.encoder_2 = DoubleConv(32, 64)
        self.pool_2 = nn.MaxPool2d(2)

        self.bridge = DoubleConv(64, 128)

        self.up_2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder_2 = DoubleConv(128, 64)

        self.up_1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decoder_1 = DoubleConv(64, 32)

        self.output_layer = nn.Conv2d(32, 1, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        encoder_1 = self.encoder_1(value)
        encoder_2 = self.encoder_2(self.pool_1(encoder_1))

        bridge = self.bridge(self.pool_2(encoder_2))

        decoder_2 = self.decoder_2(
            torch.cat([self.up_2(bridge), encoder_2], dim=1)
        )

        decoder_1 = self.decoder_1(
            torch.cat([self.up_1(decoder_2), encoder_1], dim=1)
        )

        return self.output_layer(decoder_1)


class TinyViTEncoder(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()

        self.patch_embedding = nn.Conv2d(
            3,
            dim,
            kernel_size=16,
            stride=16,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embedding(image).flatten(2).transpose(1, 2)
        return self.encoder(tokens).mean(dim=1)


class QuestionTransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 64):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            dim,
            padding_idx=0,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.embedding(tokens)).mean(dim=1)


class FusionVQAModel(nn.Module):
    def __init__(self, vocab_size: int, answer_count: int):
        super().__init__()

        self.visual_encoder = TinyViTEncoder(dim=64)

        self.text_encoder = QuestionTransformerEncoder(
            vocab_size,
            dim=64,
        )

        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + 7 + 2, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, answer_count),
        )

    def forward(
        self,
        image: torch.Tensor,
        tokens: torch.Tensor,
        detection_features: torch.Tensor,
        segmentation_features: torch.Tensor,
    ) -> torch.Tensor:
        visual_embedding = self.visual_encoder(image)
        text_embedding = self.text_encoder(tokens)

        fused_features = torch.cat(
            [
                visual_embedding,
                text_embedding,
                detection_features,
                segmentation_features,
            ],
            dim=1,
        )

        return self.fusion(fused_features)


# ==========================================================
# 2.1. ARTIFACT PREFLIGHT
# ==========================================================

def validate_week03_artifacts() -> tuple[bool, str]:
    """
    Kiểm tra checkpoint U-Net và config dùng chung trước inference.
    Không load model ở đây để giao diện vẫn khởi động khi artifact còn thiếu.
    """
    checkpoint_path = unet_weight_path()

    if checkpoint_path is None:
        return (
            False,
            "Thiếu outputs/week03_segmentation/checkpoints/"
            "unet_bottle_best.pt",
        )

    try:
        config = load_postprocess_config()
    except Exception as error:
        return False, str(error)

    return (
        True,
        "U-Net + config hợp lệ "
        f"(threshold={config['threshold']:.3f}, "
        "min_component_area_model_px="
        f"{config['min_component_area_model_px']})",
    )


# ==========================================================
# 3. LOAD MODELS
# ==========================================================

def load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(
            path,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def load_yolo():
    global _YOLO_MODEL

    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    weight = yolo_weight_path()

    if weight is None:
        raise RuntimeError(
            "Chưa tìm thấy outputs/week02_detection/weights/best.pt."
        )

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Thiếu ultralytics. Chạy pip install -r requirements.txt."
        ) from error

    _YOLO_MODEL = YOLO(str(weight))

    return _YOLO_MODEL


def load_unet() -> UNetSmall:
    global _UNET_MODEL

    if _UNET_MODEL is not None:
        return _UNET_MODEL

    checkpoint_path = unet_weight_path()

    if checkpoint_path is None:
        raise RuntimeError(
            "Chưa tìm thấy checkpoint U-Net Week03."
        )

    checkpoint = load_checkpoint(checkpoint_path)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = UNetSmall().to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    _UNET_MODEL = model

    return _UNET_MODEL


def load_fusion():
    global _FUSION_MODEL

    if _FUSION_MODEL is not None:
        return _FUSION_MODEL

    checkpoint_path = fusion_weight_path()

    if checkpoint_path is None:
        raise RuntimeError(
            "Chưa tìm thấy checkpoint Fusion VQA Week04."
        )

    checkpoint = load_checkpoint(checkpoint_path)

    question_vocab = checkpoint.get("question_vocab")
    answer_vocab = checkpoint.get("answer_vocab")

    if not question_vocab or not answer_vocab:
        raise RuntimeError(
            "Checkpoint Fusion thiếu question_vocab hoặc answer_vocab."
        )

    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = FusionVQAModel(
        vocab_size=len(question_vocab),
        answer_count=len(answer_vocab),
    ).to(DEVICE)

    model.load_state_dict(state_dict)
    model.eval()

    _FUSION_MODEL = (
        model,
        question_vocab,
        answer_vocab,
    )

    return _FUSION_MODEL


# ==========================================================
# 4. TEXT, LABEL, VQA UTILITY
# ==========================================================

def draw_label(
    draw: ImageDraw.ImageDraw,
    coordinates: tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = coordinates

    line_width = max(
        2,
        int(min(x2 - x1, y2 - y1) * 0.015),
    )

    draw.rectangle(
        (x1, y1, x2, y2),
        outline=color,
        width=line_width,
    )

    label_font = font(20)

    left, top, right, bottom = draw.textbbox(
        (0, 0),
        label,
        font=label_font,
    )

    text_width = right - left
    text_height = bottom - top

    text_y = (
        y1 - text_height - 8
        if y1 > text_height + 12
        else y1 + 3
    )

    draw.rounded_rectangle(
        (
            x1,
            text_y,
            x1 + text_width + 12,
            text_y + text_height + 8,
        ),
        radius=4,
        fill=color,
    )

    draw.text(
        (x1 + 6, text_y + 3),
        label,
        font=label_font,
        fill=(10, 10, 10),
    )


def tokenize_vietnamese(text: str) -> list[str]:
    return re.findall(
        r"\b\w+\b",
        str(text).lower(),
        flags=re.UNICODE,
    )


def normalize_text(text: str) -> str:
    normalized = str(text).replace("Đ", "D").replace("đ", "d")

    normalized = unicodedata.normalize(
        "NFD",
        normalized,
    )

    normalized = "".join(
        char
        for char in normalized
        if unicodedata.category(char) != "Mn"
    )

    return normalized.lower().strip()


def is_quality_question(question: str) -> bool:
    normalized = normalize_text(question)

    patterns = (
        "san pham co dat chat luong khong",
        "san pham co dat khong",
        "co chap nhan san pham khong",
        "san pham co dat tieu chuan khong",
        "is the product acceptable",
        "is this product acceptable",
    )

    return any(pattern in normalized for pattern in patterns)


def answer_from_vocab(answer_vocab: Any, answer_id: int) -> str:
    if isinstance(answer_vocab, dict):
        if answer_id in answer_vocab:
            return str(answer_vocab[answer_id])

        if str(answer_id) in answer_vocab:
            return str(answer_vocab[str(answer_id)])

        for answer, index in answer_vocab.items():
            if index == answer_id:
                return str(answer)

    if isinstance(answer_vocab, (list, tuple)):
        if 0 <= answer_id < len(answer_vocab):
            return str(answer_vocab[answer_id])

    return "Không xác định"


def get_vocab_index(vocab: Any, item: str) -> int:
    if not vocab:
        return -1
    if isinstance(vocab, dict):
        if item in vocab:
            try:
                return int(vocab[item])
            except (TypeError, ValueError):
                pass
        for k, v in vocab.items():
            if str(v) == str(item):
                try:
                    return int(k)
                except (TypeError, ValueError):
                    pass
            if str(k) == str(item):
                try:
                    return int(v)
                except (TypeError, ValueError):
                    pass
    elif isinstance(vocab, (list, tuple)):
        try:
            return vocab.index(item)
        except ValueError:
            for idx, val in enumerate(vocab):
                if str(val) == str(item):
                    return idx
    return -1


# ==========================================================
# 5. YOLOV8 DETECTION
# ==========================================================

def detect_surface_defect(
    image: Image.Image | None,
    confidence: float,
):
    if image is None:
        return (
            None,
            pd.DataFrame(),
            "Hãy tải một ảnh bề mặt thép trước.",
        )

    if yolo_weight_path() is None:
        return (
            image,
            pd.DataFrame(),
            "Cảnh báo: Thiếu checkpoint YOLOv8 Tuần 2.",
        )

    try:
        model = load_yolo()

        image_rgb = image.convert("RGB")

        yolo_device: int | str = (
            0
            if DEVICE.type == "cuda"
            else "cpu"
        )

        result = model.predict(
            source=np.asarray(image_rgb),
            conf=float(confidence),
            device=yolo_device,
            verbose=False,
        )[0]

        annotated = image_rgb.copy()
        draw = ImageDraw.Draw(annotated)
        rows: list[dict[str, Any]] = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                score = float(box.conf.item())

                x1, y1, x2, y2 = [
                    float(value)
                    for value in box.xyxy[0].cpu().tolist()
                ]

                label_vi = NHAN_TIENG_VIET.get(
                    class_id,
                    "Không xác định",
                )

                draw_label(
                    draw,
                    (x1, y1, x2, y2),
                    f"{label_vi} {score:.2f}",
                    MAU_LOI.get(class_id, (0, 255, 0)),
                )

                rows.append(
                    {
                        "Loại lỗi": label_vi,
                        "Độ tin cậy": round(score, 3),
                        "Tọa độ x1": round(x1, 1),
                        "Tọa độ y1": round(y1, 1),
                        "Tọa độ x2": round(x2, 1),
                        "Tọa độ y2": round(y2, 1),
                    }
                )

        table = pd.DataFrame(rows)

        status = (
            "Không phát hiện lỗi nào vượt ngưỡng đã chọn."
            if table.empty
            else f"Đã phát hiện {len(table)} vùng lỗi."
        )

        return annotated, table, status

    except Exception as error:
        return image, pd.DataFrame(), f"Lỗi Detection: {error}"


# ==========================================================
# 6. U-NET POSTPROCESS MODEL-SPACE
# ==========================================================

def postprocess_probability_small(
    probability_small: np.ndarray,
    threshold: float,
    min_component_area_model_px: int,
) -> np.ndarray:
    """
    Threshold và lọc connected component ở 256x256.
    Chỉ sau đó mới resize binary mask về ảnh gốc.
    """
    probability_small = np.asarray(
        probability_small,
        dtype=np.float32,
    )

    if probability_small.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            "Probability map phải ở model-space "
            f"{IMAGE_SIZE}x{IMAGE_SIZE}, "
            f"nhưng nhận {probability_small.shape}."
        )

    binary_small = (
        probability_small >= threshold
    ).astype(np.uint8)

    if (
        min_component_area_model_px <= 0
        or binary_small.sum() == 0
    ):
        return binary_small

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_small,
        connectivity=8,
    )

    cleaned_small = np.zeros_like(binary_small)

    for label_id in range(1, num_labels):
        component_area = int(
            stats[label_id, cv2.CC_STAT_AREA]
        )

        if component_area >= min_component_area_model_px:
            cleaned_small[labels == label_id] = 1

    return cleaned_small


def predict_unet_mask(image: Image.Image) -> np.ndarray:
    """
    U-Net probability 256x256
    -> threshold + min component area tại 256x256
    -> resize binary mask đã lọc về ảnh gốc.
    """
    model = load_unet()
    config = load_postprocess_config()

    image_rgb = np.asarray(image.convert("RGB"))

    original_height, original_width = image_rgb.shape[:2]

    resized = cv2.resize(
        image_rgb,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    tensor = (
        torch.from_numpy(
            resized.transpose(2, 0, 1)
        )
        .float()
        .unsqueeze(0)
        .to(DEVICE)
        / 255.0
    )

    with torch.no_grad():
        logits = model(tensor)

        probability_small = torch.sigmoid(
            logits
        )[0, 0].cpu().numpy()

    mask_small = postprocess_probability_small(
        probability_small=probability_small,
        threshold=float(config["threshold"]),
        min_component_area_model_px=int(
            config["min_component_area_model_px"]
        ),
    )

    mask_original = cv2.resize(
        mask_small,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    )

    return (mask_original > 0).astype(np.uint8)


# ==========================================================
# 7. MASK FEATURE, CAPTION VÀ FUSION VQA
# ==========================================================

def mask_features(mask: np.ndarray):
    binary_mask = (
        np.asarray(mask) > 0
    ).astype(np.uint8)

    height, width = binary_mask.shape

    num_labels, _, statistics, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

    boxes = []

    for label_id in range(1, num_labels):
        x, y, box_width, box_height, area = statistics[label_id]

        if int(area) > 0:
            boxes.append(
                (
                    int(x),
                    int(y),
                    int(box_width),
                    int(box_height),
                    int(area),
                )
            )

    area_ratio = float(binary_mask.mean())

    if boxes:
        x, y, box_width, box_height, _ = max(
            boxes,
            key=lambda item: item[4],
        )

        detection = np.array(
            [
                len(boxes),
                x / width,
                y / height,
                box_width / width,
                box_height / height,
                area_ratio,
                area_ratio,
            ],
            dtype=np.float32,
        )
    else:
        detection = np.array(
            [0, 0, 0, 0, 0, area_ratio, area_ratio],
            dtype=np.float32,
        )

    segmentation = np.array(
        [
            area_ratio,
            float(len(boxes) > 0),
        ],
        dtype=np.float32,
    )

    return detection, segmentation, boxes


def defect_location(detection: np.ndarray) -> str:
    if int(detection[0]) == 0:
        return "Không có vị trí lỗi"

    center_x = float(
        detection[1] + detection[3] / 2
    )

    center_y = float(
        detection[2] + detection[4] / 2
    )

    columns = [
        "bên trái",
        "chính giữa",
        "bên phải",
    ]

    rows = [
        "phía trên",
        "ở giữa",
        "phía dưới",
    ]

    return (
        f"{rows[min(2, int(center_y * 3))]} - "
        f"{columns[min(2, int(center_x * 3))]}"
    )


def structured_caption(detection: np.ndarray) -> str:
    if int(detection[0]) == 0:
        return (
            "Caption có cấu trúc: Không phát hiện vùng lỗi rõ ràng. "
            "Đề xuất kiểm tra chất lượng: Chấp nhận."
        )

    return (
        f"Caption có cấu trúc: Phát hiện {int(detection[0])} vùng lỗi. "
        f"Vùng lớn nhất nằm {defect_location(detection)}; "
        f"tỷ lệ diện tích lỗi khoảng {float(detection[5]):.2%}. "
        "Đề xuất kiểm tra chất lượng: Loại bỏ."
    )


def answer_vqa_raw(
    image: Image.Image,
    question: str,
    detection: np.ndarray,
    segmentation: np.ndarray,
) -> str:
    model, question_vocab, answer_vocab = load_fusion()

    image_rgb = np.asarray(image.convert("RGB"))

    image_rgb = cv2.resize(
        image_rgb,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    token_ids = [
        question_vocab.get(token, 1)
        for token in tokenize_vietnamese(question)
    ]

    token_ids = (
        token_ids + [0] * MAX_QUESTION_LENGTH
    )[:MAX_QUESTION_LENGTH]

    image_tensor = (
        torch.from_numpy(
            image_rgb.transpose(2, 0, 1)
        )
        .float()
        .unsqueeze(0)
        .to(DEVICE)
        / 255.0
    )

    question_tensor = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    detection_tensor = torch.tensor(
        [np.asarray(detection, dtype=np.float32)],
        dtype=torch.float32,
        device=DEVICE,
    )

    segmentation_tensor = torch.tensor(
        [np.asarray(segmentation, dtype=np.float32)],
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():
        logits = model(
            image_tensor,
            question_tensor,
            detection_tensor,
            segmentation_tensor,
        )

    valid_indices = []
    if question == "Sản phẩm có lỗi bề mặt không?":
        valid_indices = [get_vocab_index(answer_vocab, "Có lỗi bề mặt"), get_vocab_index(answer_vocab, "Không phát hiện lỗi")]
    elif question == "Sản phẩm có đạt chất lượng không?":
        valid_indices = [get_vocab_index(answer_vocab, "Chấp nhận"), get_vocab_index(answer_vocab, "Loại bỏ")]
    elif question == "Vùng lỗi nằm ở vị trí nào?":
        valid_indices = [get_vocab_index(answer_vocab, ans) for ans in [
            "Không có vị trí lỗi",
            "phía trên - bên trái", "phía trên - chính giữa", "phía trên - bên phải",
            "ở giữa - bên trái", "ở giữa - chính giữa", "ở giữa - bên phải",
            "phía dưới - bên trái", "phía dưới - chính giữa", "phía dưới - bên phải"
        ]]
    elif question == "Diện tích vùng lỗi lớn hay nhỏ?":
        valid_indices = [get_vocab_index(answer_vocab, "Diện tích lỗi lớn"), get_vocab_index(answer_vocab, "Diện tích lỗi nhỏ")]
    elif question == "Có bao nhiêu vùng lỗi?":
        valid_indices = [get_vocab_index(answer_vocab, "Không có vùng lỗi"), get_vocab_index(answer_vocab, "Một vùng lỗi"), get_vocab_index(answer_vocab, "Nhiều vùng lỗi")]
    elif question == "Có cần loại bỏ sản phẩm không?":
        valid_indices = [get_vocab_index(answer_vocab, "Chấp nhận"), get_vocab_index(answer_vocab, "Loại bỏ")]
        
    valid_indices = [idx for idx in valid_indices if idx != -1]
    
    if valid_indices:
        mask = torch.zeros(logits.size(1), dtype=torch.bool, device=DEVICE)
        mask[valid_indices] = True
        logits = logits.masked_fill(~mask, -1e9)

    answer_id = int(logits.argmax(dim=1).item())

    return answer_from_vocab(
        answer_vocab,
        answer_id,
    )


# ==========================================================
# 8. TẠO 3 PANEL CHO SEGMENTATION/VQA
# ==========================================================

def make_segmentation_preview(
    image: Image.Image,
    mask: np.ndarray,
    boxes: list[tuple[int, int, int, int, int]],
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))

    mask_binary = (mask > 0).astype(np.uint8)

    mask_visual = np.zeros_like(rgb)
    mask_visual[mask_binary > 0] = (255, 255, 255)

    overlay = rgb.copy()
    overlay[mask_binary > 0] = (255, 70, 70)

    overlay = cv2.addWeighted(
        rgb,
        0.62,
        overlay,
        0.38,
        0,
    )

    overlay_pil = Image.fromarray(overlay)
    draw = ImageDraw.Draw(overlay_pil)

    for x, y, width, height, _ in boxes:
        draw_label(
            draw,
            (x, y, x + width, y + height),
            "Vùng lỗi",
            (255, 210, 0),
        )

    panels = [
        (Image.fromarray(rgb), "Ảnh sản phẩm"),
        (Image.fromarray(mask_visual), "Mask U-Net dự đoán"),
        (overlay_pil, "Overlay vùng lỗi"),
    ]

    panel_width = max(panel.width for panel, _ in panels)
    panel_height = max(panel.height for panel, _ in panels)

    title_height = 42
    separator = 12

    canvas = Image.new(
        "RGB",
        (
            panel_width * len(panels)
            + separator * (len(panels) - 1),
            panel_height + title_height,
        ),
        color=(255, 255, 255),
    )

    canvas_draw = ImageDraw.Draw(canvas)
    title_font = font(20)

    for index, (panel, title) in enumerate(panels):
        x_offset = index * (panel_width + separator)
        y_offset = title_height

        if panel.size != (panel_width, panel_height):
            panel = panel.resize(
                (panel_width, panel_height),
                Image.Resampling.LANCZOS,
            )

        canvas.paste(panel, (x_offset, y_offset))

        left, _, right, _ = canvas_draw.textbbox(
            (0, 0),
            title,
            font=title_font,
        )

        text_x = x_offset + (
            panel_width - (right - left)
        ) // 2

        canvas_draw.text(
            (text_x, 10),
            title,
            font=title_font,
            fill=(25, 25, 25),
        )

    return canvas


# ==========================================================
# 9. QUALITY INSPECTION VÀ VQA
# ==========================================================

def segmentation_vqa(
    image: Image.Image | None,
    question: str,
):
    if image is None:
        warning_html = """
        <div class="report-card" style="border-color: #fef08a; background: #fefce8; padding: 16px;">
            <div style="color: #854d0e; font-weight: 600; text-align: center; font-size: 0.95rem;">
                ⚠️ Hãy tải ảnh MVTec AD bottle hoặc chọn ảnh mẫu trước.
            </div>
        </div>
        """
        return None, warning_html

    try:
        config = load_postprocess_config()

        if unet_weight_path() is None:
            raise RuntimeError(
                "Thiếu checkpoint U-Net Week03. "
                "Hãy import artifact Kaggle mới."
            )

        if fusion_weight_path() is None:
            raise RuntimeError(
                "Thiếu checkpoint Fusion VQA Week04. "
                "Hãy import artifact Kaggle mới."
            )

        mask = predict_unet_mask(image)

        detection, segmentation, boxes = mask_features(mask)

        quality_decision = (
            "Loại bỏ"
            if int(detection[0]) > 0
            else "Chấp nhận"
        )

        vqa_answer_raw = answer_vqa_raw(
            image,
            question,
            detection,
            segmentation,
        )

        if is_quality_question(question):
            vqa_answer_displayed = quality_decision
            consistency_rule = "segmentation_safety_gate"
        else:
            vqa_answer_displayed = vqa_answer_raw
            consistency_rule = "fusion_raw_answer"

        preview = make_segmentation_preview(
            image,
            mask,
            boxes,
        )

        badge_class = "badge-accept" if quality_decision == "Chấp nhận" else "badge-reject"
        badge_text = f"🟢 {quality_decision.upper()}" if quality_decision == "Chấp nhận" else f"🔴 {quality_decision.upper()}"
        
        report_html = f"""
        <div class="report-card">
            <div class="report-header">
                <h3 class="report-title">📋 KẾT QUẢ KIỂM TRA CHẤT LƯỢNG</h3>
                <span class="{badge_class}">{badge_text}</span>
            </div>
            
            <div class="caption-box">
                <strong>📝 Mô tả chi tiết (Caption):</strong><br>
                {structured_caption(detection)}
            </div>
            
            <div class="report-grid">
                <!-- VQA Section -->
                <div class="report-section">
                    <h4>🧠 Kết quả hỏi đáp VQA</h4>
                    <div class="report-item">
                        <span class="report-item-label">Câu hỏi:</span>
                        <span class="report-item-value">{question}</span>
                    </div>
                    <div class="report-item">
                        <span class="report-item-label">VQA hiển thị:</span>
                        <span class="report-item-value" style="color: #0f766e; font-weight: 800;">{vqa_answer_displayed}</span>
                    </div>
                    <div class="report-item">
                        <span class="report-item-label">VQA thô (logits):</span>
                        <span class="report-item-value">{vqa_answer_raw}</span>
                    </div>
                    <div class="report-item">
                        <span class="report-item-label">Quy tắc đồng bộ:</span>
                        <span class="report-item-value" style="font-family: monospace; font-size: 0.8rem; background: #e2e8f0; padding: 2px 6px; border-radius: 4px;">{consistency_rule}</span>
                    </div>
                </div>
                
                <!-- Defect Metrics Section -->
                <div class="report-section">
                    <h4>📐 Đặc trưng hình học lỗi</h4>
                    <div class="report-item">
                        <span class="report-item-label">Số vùng lỗi:</span>
                        <span class="report-item-value">{int(detection[0])}</span>
                    </div>
                    <div class="report-item">
                        <span class="report-item-label">Vị trí lỗi chính:</span>
                        <span class="report-item-value">{defect_location(detection)}</span>
                    </div>
                    <div class="report-item">
                        <span class="report-item-label">Tỷ lệ diện tích lỗi:</span>
                        <span class="report-item-value">{float(detection[5]):.2%}</span>
                    </div>
                    <div class="report-item">
                        <span class="report-item-label">Không gian hậu xử lý:</span>
                        <span class="report-item-value">Model-space (256x256)</span>
                    </div>
                </div>
            </div>
            
            <!-- Model params -->
            <div style="font-size: 0.8rem; color: #94a3b8; text-align: right; border-top: 1px solid #e2e8f0; padding-top: 8px; margin-top: 8px;">
                Nguồn mask: <strong>U-Net dự đoán</strong> &nbsp;|&nbsp; 
                Ngưỡng: <strong>{config['threshold']:.3f}</strong> &nbsp;|&nbsp; 
                Diện tích tối thiểu: <strong>{config['min_component_area_model_px']} px</strong>
            </div>
        </div>
        """

        return preview, report_html

    except Exception as error:
        error_html = f"""
        <div class="report-card" style="border-color: #fecaca; background: #fef2f2;">
            <div class="report-header" style="border-bottom-color: #fee2e2;">
                <h3 class="report-title" style="color: #dc2626;">🚨 Lỗi hệ thống</h3>
            </div>
            <div style="color: #991b1b; font-size: 0.9rem; font-family: monospace; white-space: pre-wrap; margin-top: 8px;">
                {error}
            </div>
        </div>
        """
        return image, error_html


# ==========================================================
# 10. STATUS VÀ GRADIO UI
# ==========================================================

def runtime_status() -> str:
    yolo_status = (
        "Sẵn sàng"
        if yolo_weight_path()
        else "Chưa có best.pt Tuần 2"
    )

    week03_ready, week03_message = validate_week03_artifacts()

    unet_status = (
        "Sẵn sàng"
        if week03_ready
        else "Chưa sẵn sàng"
    )

    fusion_status = (
        "Sẵn sàng"
        if fusion_weight_path()
        else "Chưa có checkpoint Fusion VQA Tuần 4"
    )

    gpu_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )

    try:
        config = load_postprocess_config()

        config_status = (
            f"threshold={config['threshold']:.3f}, "
            "min_component_area_model_px="
            f"{config['min_component_area_model_px']}"
        )
    except Exception as error:
        config_status = f"Chưa hợp lệ: {error}"

    return (
        f"**Thiết bị:** {gpu_name}  \n"
        f"**YOLOv8 Detection:** {yolo_status}  \n"
        f"**U-Net Segmentation:** {unet_status}  \n"
        f"**Week03 artifact:** {week03_message}  \n"
        f"**Fusion VQA:** {fusion_status}  \n"
        f"**Postprocess Week03:** {config_status}"
    )


def runtime_status_html() -> str:
    # 1. Device check
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "CPU"
    device_badge = f'<span class="status-ready">🟢 {gpu_name}</span>' if gpu_available else '<span class="status-ready" style="color: #64748b; background: #f1f5f9;">⚪ CPU</span>'

    # 2. YOLOv8
    has_yolo = yolo_weight_path() is not None
    yolo_badge = '<span class="status-ready">🟢 Sẵn sàng</span>' if has_yolo else '<span class="status-not-ready">🔴 Thiếu model</span>'

    # 3. U-Net
    week03_ready, week03_message = validate_week03_artifacts()
    unet_badge = '<span class="status-ready">🟢 Sẵn sàng</span>' if week03_ready else '<span class="status-not-ready">🔴 Chưa sẵn sàng</span>'

    # 4. Fusion VQA
    has_fusion = fusion_weight_path() is not None
    fusion_badge = '<span class="status-ready">🟢 Sẵn sàng</span>' if has_fusion else '<span class="status-not-ready">🔴 Thiếu checkpoint</span>'

    # 5. Config Week 03
    try:
        config = load_postprocess_config()
        config_val = f"Threshold: {config['threshold']:.3f} | Min Area: {config['min_component_area_model_px']} px"
        config_badge = f'<span class="status-ready">🟢 {config_val}</span>'
    except Exception as error:
        config_badge = f'<span class="status-not-ready">🔴 Lỗi: {error}</span>'

    html = f"""
    <div class="status-grid">
        <div class="status-item">
            <div class="status-label">Thiết bị xử lý</div>
            <div class="status-value">{device_badge}</div>
        </div>
        <div class="status-item">
            <div class="status-label">Mô hình YOLOv8</div>
            <div class="status-value">{yolo_badge}</div>
        </div>
        <div class="status-item">
            <div class="status-label">Mô hình U-Net</div>
            <div class="status-value">{unet_badge}</div>
        </div>
        <div class="status-item">
            <div class="status-label">Mô hình Fusion VQA</div>
            <div class="status-value">{fusion_badge}</div>
        </div>
        <div class="status-item" style="grid-column: span 2;">
            <div class="status-label">Cấu hình hậu xử lý (Week 03)</div>
            <div class="status-value">{config_badge}</div>
        </div>
    </div>
    """
    return html


CUSTOM_CSS = """
.container {
    max-width: 1200px;
    margin: 0 auto;
}
.header-banner {
    background: linear-gradient(135deg, #0d9488 0%, #1e1b4b 100%);
    color: white;
    padding: 35px 20px;
    border-radius: 16px;
    margin-bottom: 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(13, 148, 136, 0.15);
}
.header-banner h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0 0 10px 0;
    color: #f0fdfa;
    letter-spacing: -0.025em;
}
.header-banner p {
    font-size: 1.05rem;
    color: #ccfbf1;
    margin: 0;
    font-weight: 400;
}
.status-card {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    background: #f8fafc;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
    margin-bottom: 24px;
}
.status-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
}
.status-item {
    background: white;
    padding: 12px 16px;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.01);
}
.status-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: #64748b;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.status-value {
    font-size: 0.95rem;
    font-weight: 600;
}
.status-ready {
    color: #059669;
    background: #ecfdf5;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
}
.status-not-ready {
    color: #dc2626;
    background: #fef2f2;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
}
.tab-header {
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 16px;
    padding-bottom: 8px;
}
.tab-header h2 {
    font-size: 1.4rem;
    color: #0f766e;
    margin: 0;
    font-weight: 700;
}
.tab-header p {
    font-size: 0.95rem;
    color: #64748b;
    margin: 4px 0 0 0;
}
.btn-primary {
    background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 10px 20px !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 12px rgba(13, 148, 136, 0.2) !important;
    transition: all 0.2s ease !important;
    cursor: pointer;
}
.btn-primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(13, 148, 136, 0.30) !important;
}
.status-box {
    background: #f8fafc;
    border-left: 4px solid #0d9488;
    padding: 12px;
    border-radius: 4px;
    font-weight: 500;
}
.quick-questions-title {
    font-weight: 600;
    color: #334155;
    margin-bottom: 6px;
    font-size: 0.9rem;
}
.quick-questions-container {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 14px;
}
.question-chip {
    font-size: 0.82rem !important;
    padding: 4px 10px !important;
    background: #f1f5f9 !important;
    color: #475569 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 16px !important;
    transition: all 0.15s ease !important;
    cursor: pointer;
}
.question-chip:hover {
    background: #e2e8f0 !important;
    color: #0f766e !important;
    border-color: #94a3b8 !important;
}
.vqa-results-box {
    background: #fdfdfd;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.02);
}
.report-card {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
}
.report-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 12px;
    margin-bottom: 16px;
}
.report-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #0f766e;
    margin: 0;
}
.badge-accept {
    color: #059669;
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    padding: 6px 16px;
    border-radius: 9999px;
    font-size: 0.9rem;
    font-weight: 700;
    text-transform: uppercase;
}
.badge-reject {
    color: #dc2626;
    background: #fef2f2;
    border: 1px solid #fecaca;
    padding: 6px 16px;
    border-radius: 9999px;
    font-size: 0.9rem;
    font-weight: 700;
    text-transform: uppercase;
}
.report-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 16px;
    margin-bottom: 16px;
}
.report-section {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 14px;
}
.report-section h4 {
    margin: 0 0 10px 0;
    font-size: 0.85rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 700;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 4px;
}
.report-item {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    font-size: 0.9rem;
    line-height: 1.4;
}
.report-item:last-child {
    margin-bottom: 0;
}
.report-item-label {
    color: #64748b;
    font-weight: 500;
    white-space: nowrap;
    flex-shrink: 0;
}
.report-item-value {
    color: #1e293b;
    font-weight: 600;
    text-align: right;
    margin-left: 8px;
    word-break: break-word;
}
.caption-box {
    background: #f0fdfa;
    border-left: 4px solid #0d9488;
    padding: 12px;
    border-radius: 4px;
    margin-bottom: 16px;
    color: #0f766e;
    font-size: 0.92rem;
    line-height: 1.5;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="indigo",
        neutral_hue="slate",
    ),
    css=CUSTOM_CSS,
    title="Hệ thống phát hiện và phân vùng lỗi sản phẩm công nghiệp",
) as demo:
    
    with gr.Column(elem_classes=["container"]):
        # 1. Header Banner
        gr.HTML(
            """
            <div class="header-banner">
              <h1>Hệ thống phát hiện và phân vùng lỗi sản phẩm công nghiệp</h1>
              <p>YOLOv8 Object Detection &nbsp;|&nbsp; U-Net Pixel-level Segmentation &nbsp;|&nbsp; Fusion Multi-modal VQA &nbsp;|&nbsp; Quality Gate Safety System</p>
            </div>
            """
        )

        # 2. Main Tabbed Layout
        with gr.Tabs():
            # TAB 1: YOLOv8 Surface Defect Detection
            with gr.Tab("Nhận dạng lỗi bề mặt thép"):
                gr.HTML(
                    """
                    <div class="tab-header">
                      <h2>Phát hiện lỗi bề mặt (YOLOv8 Detection)</h2>
                      <p>Nhận dạng 6 loại lỗi bề mặt thép (Nứt rạn, Tạp chất lẫn, Vết vá, Rỗ bề mặt, Vảy cán, Trầy xước) từ tập dữ liệu NEU-DET.</p>
                    </div>
                    """
                )

                with gr.Row():
                    with gr.Column(scale=4):
                        detection_input = gr.Image(
                            type="pil",
                            label="Tải lên ảnh bề mặt thép cần kiểm tra",
                        )

                        confidence = gr.Slider(
                            minimum=0.05,
                            maximum=0.95,
                            value=0.25,
                            step=0.05,
                            label="Ngưỡng tin cậy (Confidence Threshold)",
                        )

                        detection_button = gr.Button(
                            "Bắt đầu nhận dạng",
                            variant="primary",
                            elem_classes=["btn-primary"],
                        )

                    with gr.Column(scale=5):
                        detection_output = gr.Image(
                            type="pil",
                            label="Kết quả nhận dạng (BBoxes nhãn Tiếng Việt)",
                        )

                        detection_status = gr.Textbox(
                            label="Kết luận sơ bộ",
                            interactive=False,
                            elem_classes=["status-box"],
                        )
                
                # Defect List Dataframe
                gr.Markdown("### Kết quả chi tiết")
                detection_table = gr.Dataframe(
                    headers=[
                        "Loại lỗi",
                        "Độ tin cậy",
                        "Tọa độ x1",
                        "Tọa độ y1",
                        "Tọa độ x2",
                        "Tọa độ y2",
                    ],
                    label=None,
                    interactive=False,
                )

                # Click-to-Run Examples for YOLOv8
                gr.Markdown("### Ảnh mẫu chạy thử")
                gr.Examples(
                    examples=[
                        ["datasets/detection_yolov8_neu/data/NEU-DET/test/images/scratches_271.jpg", 0.25],
                        ["datasets/detection_yolov8_neu/data/NEU-DET/test/images/patches_271.jpg", 0.25],
                        ["datasets/detection_yolov8_neu/data/NEU-DET/test/images/crazing_271.jpg", 0.25],
                        ["datasets/detection_yolov8_neu/data/NEU-DET/test/images/pitted_surface_271.jpg", 0.25],
                        ["datasets/detection_yolov8_neu/data/NEU-DET/test/images/rolled-in_scale_271.jpg", 0.25],
                        ["datasets/detection_yolov8_neu/data/NEU-DET/test/images/inclusion_271.jpg", 0.25],
                    ],
                    inputs=[detection_input, confidence],
                    outputs=[
                        detection_output,
                        detection_table,
                        detection_status,
                    ],
                    fn=detect_surface_defect,
                    cache_examples=False,
                )

                # Link function click
                detection_button.click(
                    detect_surface_defect,
                    inputs=[detection_input, confidence],
                    outputs=[
                        detection_output,
                        detection_table,
                        detection_status,
                    ],
                )

            # TAB 2: U-Net Segmentation & Fusion VQA
            with gr.Tab("Phân vùng lỗi chai lọ & Trả lời VQA"):
                gr.HTML(
                    """
                    <div class="tab-header">
                      <h2>Phân vùng lỗi & Hỏi đáp Đa phương thức (U-Net & Fusion VQA)</h2>
                      <p>Phân vùng lỗi mức pixel trên chai lọ (MVTec bottle). VQA suy luận kết hợp đặc trưng hình ảnh và ngữ cảnh câu hỏi, bảo vệ bằng Quality Safety Gate.</p>
                    </div>
                    """
                )

                with gr.Row():
                    with gr.Column(scale=5):
                        segmentation_input = gr.Image(
                            type="pil",
                            label="Tải lên ảnh chai lọ (MVTec AD Bottle)",
                        )

                        # Question Input & Quick Question Chips
                        with gr.Group():
                            question = gr.Textbox(
                                label="Nhập câu hỏi tự do hoặc chọn gợi ý bên dưới",
                                value="Sản phẩm có đạt chất lượng không?",
                                placeholder="Nhập câu hỏi tiếng Việt tại đây...",
                            )
                            
                            gr.HTML('<div class="quick-questions-title">Câu hỏi gợi ý:</div>')
                            with gr.Row(elem_classes=["quick-questions-container"]):
                                q_btn1 = gr.Button("Sản phẩm có đạt chất lượng không?", elem_classes=["question-chip"], size="sm")
                                q_btn2 = gr.Button("Sản phẩm có lỗi bề mặt không?", elem_classes=["question-chip"], size="sm")
                                q_btn3 = gr.Button("Vùng lỗi nằm ở vị trí nào?", elem_classes=["question-chip"], size="sm")
                                q_btn4 = gr.Button("Có bao nhiêu vùng lỗi?", elem_classes=["question-chip"], size="sm")
                                q_btn5 = gr.Button("Diện tích vùng lỗi lớn hay nhỏ?", elem_classes=["question-chip"], size="sm")
                                q_btn6 = gr.Button("Có cần loại bỏ sản phẩm không?", elem_classes=["question-chip"], size="sm")

                        segmentation_button = gr.Button(
                            "Bắt đầu phân tích",
                            variant="primary",
                            elem_classes=["btn-primary"],
                        )

                    with gr.Column(scale=7):
                        segmentation_output = gr.Image(
                            type="pil",
                            label="Bảng so sánh: Ảnh gốc | Mask U-Net dự đoán | Overlay vùng lỗi thực tế",
                        )

                        with gr.Group(elem_classes=["vqa-results-box"]):
                            vqa_report = gr.HTML()

                # Event handlers for Quick Question Chips
                q_btn1.click(lambda: "Sản phẩm có đạt chất lượng không?", outputs=question)
                q_btn2.click(lambda: "Sản phẩm có lỗi bề mặt không?", outputs=question)
                q_btn3.click(lambda: "Vùng lỗi nằm ở vị trí nào?", outputs=question)
                q_btn4.click(lambda: "Có bao nhiêu vùng lỗi?", outputs=question)
                q_btn5.click(lambda: "Diện tích vùng lỗi lớn hay nhỏ?", outputs=question)
                q_btn6.click(lambda: "Có cần loại bỏ sản phẩm không?", outputs=question)

                # Click-to-Run Examples for Tab 2
                gr.Markdown("### Ảnh mẫu chạy thử")
                gr.Examples(
                    examples=[
                        ["datasets/sample_images/broken_bottle_sample.png", "Sản phẩm có đạt chất lượng không?"],
                        ["datasets/sample_images/broken_bottle_sample.png", "Có bao nhiêu vùng lỗi?"],
                        ["datasets/sample_images/broken_bottle_sample.png", "Vùng lỗi nằm ở vị trí nào?"],
                        ["datasets/sample_images/good_bottle_sample.png", "Sản phẩm có đạt chất lượng không?"],
                        ["datasets/sample_images/good_bottle_sample.png", "Sản phẩm có lỗi bề mặt không?"],
                    ],
                    inputs=[segmentation_input, question],
                    outputs=[
                        segmentation_output,
                        vqa_report,
                    ],
                    fn=segmentation_vqa,
                    cache_examples=False,
                )

                # Link function click
                segmentation_button.click(
                    segmentation_vqa,
                    inputs=[segmentation_input, question],
                    outputs=[
                        segmentation_output,
                        vqa_report,
                    ],
                )

            # TAB 3: System Status & User Guide
            with gr.Tab("Cấu hình & Hướng dẫn"):
                gr.HTML(
                    """
                    <div class="tab-header">
                      <h2>Trạng thái hệ thống & Hướng dẫn sử dụng</h2>
                      <p>Thông tin cấu hình phần cứng, tham số hậu xử lý và trạng thái hoạt động của các mô hình.</p>
                    </div>
                    """
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### Trạng thái tài nguyên & Checkpoint")
                        # Display styled HTML status dashboard
                        status_html_output = gr.HTML(runtime_status_html())
                        
                        refresh_status_btn = gr.Button("Cập nhật trạng thái", size="sm")
                        refresh_status_btn.click(runtime_status_html, outputs=status_html_output)
                        
                    with gr.Column(scale=4):
                        gr.Markdown(
                            """
                            ### Hướng dẫn sử dụng nhanh
                            
                            **1. Nhận dạng lỗi bề mặt thép (YOLOv8)**
                            - Chọn tab **"Nhận dạng lỗi bề mặt thép"**.
                            - Tải ảnh lên hoặc bấm chọn một trong các **Ảnh mẫu chạy thử** ở dưới.
                            - Kéo slider để điều chỉnh ngưỡng tin cậy (Confidence).
                            - Bấm **Bắt đầu nhận dạng** để xem ảnh vẽ bounding box lỗi và bảng thông số chi tiết.
                            
                            **2. Phân vùng lỗi & Hỏi đáp VQA chai lọ (U-Net + VQA)**
                            - Chọn tab **"Phân vùng lỗi chai lọ & Trả lời VQA"**.
                            - Tải ảnh chai lọ lên hoặc chọn **Ảnh mẫu chạy thử** có sẵn.
                            - Nhập câu hỏi cần hỏi vào ô nhập liệu (hoặc click chọn nhanh một trong các câu hỏi gợi ý ở dưới).
                            - Bấm **Bắt đầu phân tích** để hiển thị ảnh phân vùng (3 panel) và bảng kết quả chất lượng.
                            
                            **3. Cơ chế đồng bộ quyết định (Safety Gate)**
                            - Khi người dùng hỏi các câu hỏi đánh giá chất lượng sản phẩm (ví dụ: *"Sản phẩm có đạt chất lượng không?"*), hệ thống sẽ tự động kích hoạt Safety Gate để đồng bộ câu trả lời hiển thị dựa trên kết quả phân vùng lỗi thực tế của mô hình U-Net. Nếu phát hiện diện tích lỗi vượt ngưỡng cấu hình, quyết định cuối cùng luôn là **"Loại bỏ"**, ngược lại là **"Chấp nhận"**. Quy trình này giúp loại bỏ hoàn toàn sự không nhất quán giữa kết quả phân vùng và câu trả lời VQA sinh ra.
                            """
                        )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )