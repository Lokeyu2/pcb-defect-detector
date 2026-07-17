# PCB Defect Detection

基于YOLOv8的PCB缺陷检测系统，支持PC端推理和地平线RDK X5边缘端部署。

## Defect Classes

| Index | Class | Description |
|:-----:|:------|:------------|
| 0 | open | 开路/断线 |
| 1 | short | 短路 |
| 2 | mousebite | 鼠咬缺角 |
| 3 | spur | 毛刺/突铜 |
| 4 | copper | 铜箔异常 |
| 5 | pin-hole | 针孔 |

## Project Structure

```
├─ model/                Model weights
│  ├─ best.pt            YOLOv8 trained weights
│  ├─ best.onnx          ONNX export
│  └─ bin_output/        Horizon BIN (for RDK X5)
├─ config/               ONNX->BIN config YAML
├─ data/
│  └─ test_images/       Test images
├─ scripts/              Scripts
│  ├─ train.py           Train
│  ├─ export.py          Export ONNX
│  ├─ detect.py          PC inference
│  ├─ evaluate.py        Batch evaluation
│  └─ convert_bin.py     ONNX->BIN conversion
├─ results/              Output directory
└─ 实验日志.md            Experiment log (Chinese)
```

## Quick Start

### PC Inference

```bash
python scripts/detect.py                    # all test images
python scripts/detect.py --source path/to/image.jpg
python scripts/detect.py --source 0          # webcam
python scripts/detect.py --conf 0.7          # custom threshold
python scripts/detect.py --no-rules          # disable post-processing
```

### Evaluate

```bash
python scripts/evaluate.py
```

### Export ONNX

```bash
python scripts/export.py
```

### Convert to Horizon BIN

```bash
# Docker Desktop must be running
python scripts/convert_bin.py
```

## Results

| Metric | Value |
|:-------|:------|
| mAP50 | 0.991 |
| Precision | 0.982 |
| Recall | 0.971 |
| Model | YOLOv8n |
| Input | 640x640 |

## Requirements

- Python 3.10+
- ultralytics
- onnxruntime
- opencv-python

## Dataset

[DeepPCB](https://github.com/Charmve/DeepPCB) — 1500 PCB image pairs with 6 defect types.
