"""
续训 pcb_v2 合并数据集模型
- 从 runs/runs/pcb_v2/weights/last.pt 续训到 180 轮
- 使用 GPU (RTX 4050)
- batch 保持 8 (按用户要求)
用法: python scripts/train_pcb_v2_resume.py
"""
from ultralytics import YOLO

if __name__ == '__main__':
    # 从上次保存的 last.pt 续训（当前100轮 → 续到180轮）
    model = YOLO(r'../runs/runs/pcb_v2/weights/last.pt')

    model.train(
        resume=True,          # 接着上次断点续训
        imgsz=640,
        batch=8,              # 保持 8，不改
        device=0,             # GPU (RTX 4050)
        workers=2,
        amp=True,
        project='../runs',
        name='pcb_v2',
        exist_ok=True,
        patience=30,          # 连续30轮无提升早停
        epochs=180,           # 总轮数，续训会自动从101补到180
        close_mosaic=10,
        val=True,             # 每轮验证，监控mAP
        plots=True,           # 生成训练曲线
    )

    # 训练完做一次完整验证
    val_results = model.val(data='../data/data.yaml', batch=8, device=0)
    print(f"\n训练完成!")
    print(f"最佳模型: runs/runs/pcb_v2/weights/best.pt")
    print(f"验证 mAP50   = {val_results.box.map50:.4f}")
    print(f"验证 mAP50-95 = {val_results.box.map:.4f}")
