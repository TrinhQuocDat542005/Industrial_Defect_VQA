from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

GATE_SUFFIXES = (
    "test/broken_large/000.png",
    "test/good/000.png",
)


def normalize_path(value: Any) -> str:
    return str(value).replace("\\", "/").strip().lower()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"[FAIL] Không đọc được JSON {display_path(path)}: {error}")
        return None

    if not isinstance(content, dict):
        print(f"[FAIL] JSON không có cấu trúc object: {display_path(path)}")
        return None

    return content


def check_file(name: str, path: Path) -> bool:
    if path.is_file():
        print(f"[PASS] {name}: {display_path(path)}")
        return True

    print(f"[FAIL] Thiếu {name}: {display_path(path)}")
    return False


def close_number(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    try:
        return math.isclose(
            float(left),
            float(right),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    except (TypeError, ValueError):
        return False


def validate_week03_config(config_path: Path) -> tuple[bool, dict[str, Any] | None]:
    config = read_json(config_path)

    if config is None:
        return False, None

    required = {
        "threshold",
        "min_component_area_model_px",
        "postprocess_space",
        "reference_image_size",
        "selection_status",
    }

    missing = sorted(required - set(config))

    if missing:
        print(
            "[FAIL] Week03 config thiếu key: "
            + ", ".join(missing)
        )
        return False, None

    try:
        threshold = float(config["threshold"])
        min_area = int(config["min_component_area_model_px"])
        reference_size = list(config["reference_image_size"])
    except (TypeError, ValueError):
        print("[FAIL] Week03 config có giá trị không đúng kiểu.")
        return False, None

    conditions = [
        (
            0.0 <= threshold <= 1.0,
            "threshold phải nằm trong [0, 1].",
        ),
        (
            min_area > 0,
            "min_component_area_model_px phải lớn hơn 0.",
        ),
        (
            str(config["postprocess_space"]).strip().lower()
            == "model_resolution",
            "postprocess_space phải là model_resolution.",
        ),
        (
            reference_size == [256, 256],
            "reference_image_size phải là [256, 256].",
        ),
        (
            str(config["selection_status"]).strip().lower()
            == "valid",
            "selection_status phải là valid.",
        ),
    ]

    all_valid = True

    for condition, message in conditions:
        if not condition:
            print(f"[FAIL] Week03 config: {message}")
            all_valid = False

    if all_valid:
        print(
            "[PASS] Week03 config hợp lệ: "
            f"threshold={threshold:.3f}, "
            f"min_component_area_model_px={min_area}, "
            "postprocess_space=model_resolution."
        )

        return True, {
            "threshold": threshold,
            "min_component_area_model_px": min_area,
        }

    return False, None


def validate_predicted_feature_csv(
    path: Path,
    expected: dict[str, Any],
) -> bool:
    try:
        frame = pd.read_csv(path)
    except Exception as error:
        print(f"[FAIL] Không đọc được qa_image_features.csv: {error}")
        return False

    required_columns = {
        "image_path",
        "source_of_mask",
        "mask_threshold",
        "min_component_area_model_px",
        "postprocess_space",
        "num_regions",
        "defect_area_ratio",
        "mask_coverage",
    }

    missing = sorted(required_columns - set(frame.columns))

    if missing:
        print(
            "[FAIL] qa_image_features.csv thiếu cột: "
            + ", ".join(missing)
        )

        if "min_component_area" in frame.columns:
            print(
                "[INFO] Phát hiện cột cũ min_component_area. "
                "Hãy chạy lại Week04 để tạo artifact mới."
            )

        return False

    if frame.empty:
        print("[FAIL] qa_image_features.csv rỗng.")
        return False

    passed = True

    if not frame["source_of_mask"].eq("unet_predicted").all():
        print(
            "[FAIL] qa_image_features.csv không dùng hoàn toàn "
            "source_of_mask=unet_predicted."
        )
        passed = False

    if not frame["postprocess_space"].eq("model_resolution").all():
        print(
            "[FAIL] qa_image_features.csv có postprocess_space không phải "
            "model_resolution."
        )
        passed = False

    threshold_ok = frame["mask_threshold"].map(
        lambda value: close_number(value, expected["threshold"])
    ).all()

    if not threshold_ok:
        print(
            "[FAIL] qa_image_features.csv không dùng threshold từ "
            "best_postprocess_config.json."
        )
        passed = False

    area_ok = frame["min_component_area_model_px"].map(
        lambda value: int(value) == int(
            expected["min_component_area_model_px"]
        )
    ).all()

    if not area_ok:
        print(
            "[FAIL] qa_image_features.csv không dùng "
            "min_component_area_model_px từ Week03."
        )
        passed = False

    if passed:
        print(
            "[PASS] qa_image_features.csv dùng U-Net predicted mask và "
            "config Week03 hiện hành."
        )

    return passed


def validate_split_has_no_gate_images(path: Path, label: str) -> bool:
    try:
        frame = pd.read_csv(path)
    except Exception as error:
        print(f"[FAIL] Không đọc được {label}: {error}")
        return False

    if "image_path" not in frame.columns:
        print(f"[FAIL] {label} thiếu cột image_path.")
        return False

    normalized = frame["image_path"].map(normalize_path)

    leakage = frame[
        normalized.map(
            lambda value: any(
                value.endswith(suffix)
                for suffix in GATE_SUFFIXES
            )
        )
    ]

    if leakage.empty:
        print(
            f"[PASS] {label} không chứa hai ảnh Quality Gate độc lập."
        )
        return True

    print(
        f"[FAIL] {label} có leakage Quality Gate: "
        + ", ".join(
            sorted(
                set(
                    leakage["image_path"]
                    .astype(str)
                    .tolist()
                )
            )
        )
    )

    return False


def validate_gate_json(
    path: Path,
    gate_name: str,
    expected_decision: str,
    expected_config: dict[str, Any],
) -> bool:
    data = read_json(path)

    if data is None:
        return False

    required = {
        "image_path",
        "question",
        "mask_threshold",
        "min_component_area_model_px",
        "postprocess_space",
        "source_of_mask",
        "vqa_answer_raw",
        "vqa_answer",
        "quality_decision",
        "vqa_consistency_rule",
    }

    missing = sorted(required - set(data))

    if missing:
        print(
            f"[FAIL] JSON {gate_name} thiếu key: "
            + ", ".join(missing)
        )
        return False

    expected_suffix = f"test/{gate_name}/000.png"

    checks = [
        (
            normalize_path(data["image_path"]).endswith(expected_suffix),
            f"image_path không phải {expected_suffix}.",
        ),
        (
            data["quality_decision"] == expected_decision,
            "quality_decision không đúng kỳ vọng.",
        ),
        (
            data["vqa_answer"] == expected_decision,
            "vqa_answer phải trùng Quality Decision "
            "với câu hỏi chất lượng.",
        ),
        (
            data["source_of_mask"] == "unet_predicted",
            "source_of_mask phải là unet_predicted.",
        ),
        (
            data["postprocess_space"] == "model_resolution",
            "postprocess_space phải là model_resolution.",
        ),
        (
            close_number(
                data["mask_threshold"],
                expected_config["threshold"],
            ),
            "mask_threshold không khớp Week03 config.",
        ),
        (
            int(data["min_component_area_model_px"])
            == int(expected_config["min_component_area_model_px"]),
            "min_component_area_model_px không khớp Week03 config.",
        ),
        (
            data["vqa_consistency_rule"]
            == "segmentation_safety_gate",
            "vqa_consistency_rule phải là segmentation_safety_gate.",
        ),
    ]

    passed = True

    for condition, message in checks:
        if not condition:
            print(f"[FAIL] JSON {gate_name}: {message}")
            passed = False

    if passed:
        print(
            f"[PASS] Quality Gate {gate_name}: "
            f"{expected_decision}; VQA hiển thị đồng bộ Quality Decision."
        )

    return passed


def validate_week04_gate_manifest(path: Path) -> bool:
    try:
        frame = pd.read_csv(path)
    except Exception as error:
        print(
            f"[FAIL] Không đọc được quality_gate_manifest_week04.csv: {error}"
        )
        return False

    required_columns = {
        "gate_name",
        "expected_decision",
        "quality_decision",
        "vqa_answer",
        "passed",
    }

    missing = sorted(required_columns - set(frame.columns))

    if missing:
        print(
            "[FAIL] quality_gate_manifest_week04.csv thiếu cột: "
            + ", ".join(missing)
        )
        return False

    expected_rows = {
        "broken_large": "Loại bỏ",
        "good": "Chấp nhận",
    }

    passed = True

    for gate_name, decision in expected_rows.items():
        subset = frame[frame["gate_name"] == gate_name]

        if len(subset) != 1:
            print(
                f"[FAIL] quality_gate_manifest_week04.csv cần đúng "
                f"một dòng cho {gate_name}."
            )
            passed = False
            continue

        row = subset.iloc[0]

        if not (
            str(row["expected_decision"]) == decision
            and str(row["quality_decision"]) == decision
            and str(row["vqa_answer"]) == decision
            and str(row["passed"]).strip().lower() in ("true", "1", "yes")
        ):
            print(
                f"[FAIL] quality_gate_manifest_week04.csv sai kết quả "
                f"cho {gate_name}."
            )
            passed = False

    if passed:
        print("[PASS] quality_gate_manifest_week04.csv hợp lệ.")

    return passed


def validate_test_metrics(path: Path) -> bool:
    try:
        frame = pd.read_csv(path)
    except Exception as error:
        print(f"[FAIL] Không đọc được test_metrics_vqa.csv: {error}")
        return False

    required_columns = {"tap_du_lieu", "accuracy", "source_of_mask"}

    missing = sorted(required_columns - set(frame.columns))

    if missing:
        print(
            "[FAIL] test_metrics_vqa.csv thiếu cột: "
            + ", ".join(missing)
        )
        return False

    if frame.empty:
        print("[FAIL] test_metrics_vqa.csv rỗng.")
        return False

    if not frame["source_of_mask"].eq("unet_predicted").all():
        print(
            "[FAIL] test_metrics_vqa.csv không ghi "
            "source_of_mask=unet_predicted."
        )
        return False

    try:
        accuracy = float(frame.iloc[0]["accuracy"])
        print(f"[INFO] Accuracy VQA thô: {accuracy:.4f}")
        if accuracy < 0.60:
            print(f"[WARN] Accuracy thấp (< 0.60).")
    except Exception:
        pass

    print(
        "[PASS] test_metrics_vqa.csv tồn tại và ghi đúng nguồn mask."
    )

    return True


def main() -> int:
    paths = {
        "Week02 checkpoint YOLOv8": (
            OUTPUTS_DIR / "week02_detection" / "weights" / "best.pt"
        ),
        "Week02 metrics": (
            OUTPUTS_DIR / "week02_detection" / "metrics_yolov8.csv"
        ),
        "Week03 checkpoint U-Net": (
            OUTPUTS_DIR
            / "week03_segmentation"
            / "checkpoints"
            / "unet_bottle_best.pt"
        ),
        "Week03 config postprocess": (
            OUTPUTS_DIR
            / "week03_segmentation"
            / "best_postprocess_config.json"
        ),
        "Week03 validation postprocess search": (
            OUTPUTS_DIR
            / "week03_segmentation"
            / "validation_postprocess_search.csv"
        ),
        "Week03 quality gate manifest": (
            OUTPUTS_DIR
            / "week03_segmentation"
            / "quality_gate_manifest.csv"
        ),
        "Week04 Fusion VQA checkpoint": (
            OUTPUTS_DIR / "week04_multimodal" / "fusion_vqa.pt"
        ),
        "Week04 qa_image_features": (
            OUTPUTS_DIR / "week04_multimodal" / "qa_image_features.csv"
        ),
        "Week04 qa_image_split_manifest": (
            OUTPUTS_DIR
            / "week04_multimodal"
            / "qa_image_split_manifest.csv"
        ),
        "Week04 qa_split_manifest": (
            OUTPUTS_DIR
            / "week04_multimodal"
            / "qa_split_manifest.csv"
        ),
        "Week04 test metrics": (
            OUTPUTS_DIR / "week04_multimodal" / "test_metrics_vqa.csv"
        ),
        "Week04 JSON broken_large": (
            OUTPUTS_DIR
            / "week04_multimodal"
            / "ket_qua_kiem_tra"
            / "w4_broken_large_ket_qua_kiem_tra_tieng_viet.json"
        ),
        "Week04 PNG broken_large": (
            OUTPUTS_DIR
            / "week04_multimodal"
            / "ket_qua_kiem_tra"
            / "w4_broken_large_ket_qua_kiem_tra_tieng_viet.png"
        ),
        "Week04 JSON good": (
            OUTPUTS_DIR
            / "week04_multimodal"
            / "ket_qua_kiem_tra"
            / "w4_good_ket_qua_kiem_tra_tieng_viet.json"
        ),
        "Week04 PNG good": (
            OUTPUTS_DIR
            / "week04_multimodal"
            / "ket_qua_kiem_tra"
            / "w4_good_ket_qua_kiem_tra_tieng_viet.png"
        ),
        "Week04 quality gate manifest": (
            OUTPUTS_DIR
            / "week04_multimodal"
            / "quality_gate_manifest_week04.csv"
        ),
    }

    all_pass = True

    print("--- KIỂM TRA FILE ARTIFACT ---")

    for name, path in paths.items():
        all_pass = check_file(name, path) and all_pass

    print("\n--- KIỂM TRA WEEK03 ---")

    expected_config = None
    config_path = paths["Week03 config postprocess"]

    if config_path.is_file():
        config_pass, expected_config = validate_week03_config(config_path)
        all_pass = config_pass and all_pass
    else:
        all_pass = False

    print("\n--- KIỂM TRA WEEK04 FEATURE VÀ SPLIT ---")

    if expected_config is not None:
        feature_path = paths["Week04 qa_image_features"]

        if feature_path.is_file():
            all_pass = (
                validate_predicted_feature_csv(
                    feature_path,
                    expected_config,
                )
                and all_pass
            )
        else:
            all_pass = False

        for key, label in [
            (
                "Week04 qa_image_split_manifest",
                "qa_image_split_manifest.csv",
            ),
            (
                "Week04 qa_split_manifest",
                "qa_split_manifest.csv",
            ),
        ]:
            path = paths[key]

            if path.is_file():
                all_pass = (
                    validate_split_has_no_gate_images(path, label)
                    and all_pass
                )
            else:
                all_pass = False

        for key, gate_name, expected_decision in [
            (
                "Week04 JSON broken_large",
                "broken_large",
                "Loại bỏ",
            ),
            (
                "Week04 JSON good",
                "good",
                "Chấp nhận",
            ),
        ]:
            path = paths[key]

            if path.is_file():
                all_pass = (
                    validate_gate_json(
                        path,
                        gate_name,
                        expected_decision,
                        expected_config,
                    )
                    and all_pass
                )
            else:
                all_pass = False

    test_metrics_path = paths["Week04 test metrics"]

    if test_metrics_path.is_file():
        all_pass = validate_test_metrics(test_metrics_path) and all_pass
    else:
        all_pass = False

    gate_manifest_path = paths["Week04 quality gate manifest"]

    if gate_manifest_path.is_file():
        all_pass = (
            validate_week04_gate_manifest(gate_manifest_path)
            and all_pass
        )
    else:
        all_pass = False

    print("\n--- KẾT LUẬN ---")

    if all_pass:
        print(
            "PASS: Artifact Week02–Week04 hợp lệ. "
            "Week04 dùng config Week03 hiện hành, không leakage Quality Gate, "
            "broken_large → Loại bỏ và good → Chấp nhận."
        )
        return 0

    print(
        "FAIL: Không dùng artifact hiện tại để nộp. "
        "Hãy Restart Kernel → Run All Week04, đóng gói artifact mới, "
        "import lại project local rồi chạy script này."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
