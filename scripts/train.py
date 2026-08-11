"""
YOLOv8 PCB缺陷检测训练脚本
- 针对真实PCB板场景优化: 更强的数据增强 + 定期验证
用法: python scripts/train.py
"""
from ultralytics import YOLO

if __name__ == '__main__':
    # 从上次保存的 last.pt 续训
    model = YOLO('../runs/runs/pcb_v2/weights/last.pt')

    results = model.train(
        resume=True,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        amp=True,
        project='../runs',
        name='pcb_v2',
        exist_ok=True,
        patience=30,

        # 学习率
        lr0=0.005,
        cos_lr=True,

        # ===== 针对真实场景的增强 =====
        augment=True,
        hsv_h=0.1,        # 色相 ±10%
        hsv_s=0.7,        # 饱和度 ±70% (模拟不同光照色温)
        hsv_v=0.4,        # 明度 ±40% (模拟光照强弱变化)
        degrees=15,        # 旋转 ±15° (模拟手持角度偏差)
        translate=0.1,     # 平移 10%
        scale=0.3,         # 缩放 ±30%
        shear=5,           # 剪切 5°
        perspective=0.001, # 轻微透视变换 (模拟拍摄角度)
        flipud=0.1,        # 上下翻转 10%
        fliplr=0.5,        # 左右翻转 50%
        mosaic=0.8,        # 拼图增强 80% (让模型看到不同区域组合)
        mixup=0.2,         # 混合增强 20% (提高泛化)
        copy_paste=0.1,    # 复制粘贴增强 10% (对小缺陷有利)

        # ===== 验证策略 =====
        val=True,           # 每epoch验证 (监控mAP)
        plots=True,         # 生成训练曲线

        # ===== 类别权重 =====
        # DeepPCB类别不平衡, 让小缺陷获得更多关注
        # close_mosaic=10,    # 最后10轮关闭mosaic, 让模型稳定收敛
    )

    # 训练完再做一次完整验证
    val_results = model.val(data='../data/data.yaml', batch=8, device=0)
    print(f"\n训练完成!")
    print(f"最佳模型保存在 runs/pcb_v2/weights/best.pt")
    print(f"验证mAP50: {val_results.box.map50:.4f}")
    print(f"验证mAP50-95: {val_results.box.map:.4f}")
