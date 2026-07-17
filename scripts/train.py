"""
YOLOv8 PCB缺陷检测训练脚本
用法: python scripts/train.py
"""
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')  # 或 yolov8s.pt
    results = model.train(
        data='../data/data.yaml',
        epochs=200,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        amp=True,
        project='../runs',
        name='pcb_yolov8',
        exist_ok=True,
        patience=50,
        lr0=0.01,
        cos_lr=True,
        augment=True,
    )
    print(f"训练完成，模型保存在 runs/pcb_yolov8/weights/best.pt")
