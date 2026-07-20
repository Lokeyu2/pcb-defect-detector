"""
YOLOv8 PCB缺陷检测训练脚本
用法: python scripts/train.py
"""
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')  # 或 yolov8s.pt
    results = model.train(
        data='data/data.yaml',
        epochs=200,
        imgsz=640,
        batch=8,            # RTX4050 6GB显存, 8更稳
        device=0,
        workers=2,          # 减轻数据加载压力
        amp=True,
        project='../runs',
        name='pcb_merged_v1',
        exist_ok=True,
        patience=40,        # 早停耐心值
        lr0=0.005,          # 合并数据集较大, 学习率适当降低
        cos_lr=True,
        augment=True,
        val=False,          # 每epoch跳过验证, 加速训练
    )
    # 训练完再做一次完整验证
    val_results = model.val(data='../data/data.yaml', batch=8, device=0)
    print(f"训练完成，最佳模型保存在 runs/pcb_merged_v1/weights/best.pt")
    print(f"验证mAP50: {val_results.box.map50:.4f}")
