# Chuáº©n bá»‹ dá»¯ liá»‡u

## 1. NEU Surface Defect Database

NEU-DET Ä‘Æ°á»£c dÃ¹ng cho Week01 vÃ  Week02, phá»¥c vá»¥ phÃ¡t hiá»‡n sÃ¡u loáº¡i lá»—i bá» máº·t thÃ©p báº±ng YOLOv8.

ÄÆ°á»ng dáº«n trong project:

```text
datasets/detection_yolov8_neu/data/NEU-DET/
â”œâ”€â”€ train/
â”‚   â”œâ”€â”€ images/
â”‚   â””â”€â”€ labels/
â””â”€â”€ test/
    â”œâ”€â”€ images/
    â””â”€â”€ labels/
```

KhÃ´ng sá»­a dá»¯ liá»‡u gá»‘c trong `external/`. CÃ¡c split train, validation vÃ  test dÃ¹ng cho thÃ­ nghiá»‡m Ä‘Æ°á»£c lÆ°u trong:

```text
outputs/week02_detection/splits/
```

## 2. MVTec AD - bottle

MVTec AD category `bottle` Ä‘Æ°á»£c dÃ¹ng cho Week03 vÃ  Week04:

- Week03: phÃ¢n vÃ¹ng lá»—i má»©c pixel báº±ng U-Net.
- Week04: dÃ¹ng U-Net predicted mask Ä‘á»ƒ táº¡o Ä‘áº·c trÆ°ng, caption cÃ³ cáº¥u trÃºc, VQA vÃ  quality inspection.

Cáº¥u trÃºc cáº§n cÃ³:

```text
datasets/mvtec_ad/bottle/
â”œâ”€â”€ train/
â”‚   â””â”€â”€ good/
â”œâ”€â”€ test/
â”‚   â”œâ”€â”€ good/
â”‚   â””â”€â”€ <defect_type>/
â””â”€â”€ ground_truth/
    â””â”€â”€ <defect_type>/
```

TrÃªn Kaggle, thÃªm dataset MVTec AD cÃ³ Ä‘áº§y Ä‘á»§ `bottle/train`, `bottle/test` vÃ  `bottle/ground_truth`. Dataset lá»›n khÃ´ng cáº§n Ä‘Æ°a vÃ o ZIP source cá»§a project.

CÃ³ thá»ƒ chuáº©n bá»‹ dá»¯ liá»‡u báº±ng script:

```powershell
python scripts/tai_mvtec_bottle.py
```

hoáº·c dÃ¹ng Kaggle Input khi cháº¡y notebook trÃªn GPU.

## 3. Kiá»ƒm tra trÆ°á»›c khi huáº¥n luyá»‡n

Notebook Week03 vÃ  Week04 cáº§n cháº¡y trong mÃ´i trÆ°á»ng cÃ³ GPU. TrÆ°á»›c khi train, kiá»ƒm tra:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

Chá»‰ bÃ¡o cÃ¡o checkpoint vÃ  metric sau khi notebook cháº¡y hoÃ n chá»‰nh trÃªn dá»¯ liá»‡u thá»±c.

