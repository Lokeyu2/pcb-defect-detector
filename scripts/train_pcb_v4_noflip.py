"""
数据增强消融实验: 关闭 Flip 再训练 (对照 pcb_v4)
- 目的: 验证"关闭 Flip 提升 mAP50-95" (参考网上的 PCB-YOLOv8 项目消融结论)
- 起点: runs/pcb_v4/weights/best.pt  (与 pcb_v4 相同的起点, 保证对比公平)
- 改动: 仅 fliplr=0, flipud=0 (pcb_v4 默认 fliplr=0.5)
- 其他参数与 pcb_v4 完全一致: 640输入, batch8, 180ep, patience60, lr0=0.005
用法: python scripts/train_pcb_v4_noflip.py
"""
import os
from ultralytics import YOLO

import paths


def main():
    BASE = str(paths.PROJECT_ROOT)
    START = os.path.join(BASE, 'runs', 'pcb_v4', 'weights', 'best.pt')
    DATA = os.path.join(BASE, 'data', 'data.yaml')

    assert os.path.exists(START), f'起点权重不存在: {START}'
    assert os.path.exists(DATA), f'数据配置不存在: {DATA}'

    model = YOLO(START)
    print(f'已加载 pcb_v4 (nc=6), 训练数据: {DATA}')
    print(f'消融: fliplr=0, flipud=0 (关闭Flip)')

    results = model.train(
        data=DATA,
        epochs=180,
        batch=8,
        device=0,
        workers=2,
        amp=True,
        imgsz=640,
        lr0=0.005,
        cos_lr=True,
        patience=60,
        close_mosaic=10,
        fliplr=0.0,      # 关闭水平翻转 (对照项)
        flipud=0.0,      # 关闭垂直翻转 (对照项)
        val=True,
        plots=True,
        project=os.path.join(BASE, 'runs'),
        name='pcb_v4_noflip',
        exist_ok=True,
    )

    val = model.val(data=DATA, batch=8, device=0)
    print(f'\n消融训练完成! 最终结果:')
    print(f'  最终 mAP50   = {val.box.map50:.4f}')
    print(f'  最终 mAP50-95 = {val.box.map:.4f}')
    print('对照 pcb_v4: mAP50=0.993 / mAP50-95=0.730')


if __name__ == '__main__':
    main()
