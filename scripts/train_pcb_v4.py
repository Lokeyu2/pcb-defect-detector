"""
重建 pcb_v4 合并数据集 PCB 模型（彻底绕开 resume 坑）
- 起点: runs/runs/pcb_merged_v1/weights/best.pt  (epoch92, PCB 6类, 完好)
- 关键改动: 不用 resume=True (它复用缓存 coco8 污染训练), 改为显式加载权重 + 显式传 data
- 全绝对路径 + 新 run 名 pcb_v4, 不碰任何旧成果
用法: python D:/DeepPCB-master/scripts/train_pcb_v4.py
"""
import os
from ultralytics import YOLO


def main():
    BASE = r'D:\DeepPCB-master'
    START = os.path.join(BASE, 'runs', 'runs', 'pcb_merged_v1', 'weights', 'best.pt')
    DATA = os.path.join(BASE, 'data', 'data.yaml')

    assert os.path.exists(START), f'起点权重不存在: {START}'
    assert os.path.exists(DATA), f'数据配置不存在: {DATA}'

    # 关键: YOLO(权重路径) 加载 PCB 模型, 然后显式传 data
    model = YOLO(START)
    print(f'已加载 PCB 模型 (nc=6), 训练数据: {DATA}')
    print(f'注意: 若下面打印 nc=80/person, 说明加载失败, 立即 Ctrl+C')

    results = model.train(
        data=DATA,            # 显式传数据, 不依赖 resume 缓存
        epochs=180,           # 全新训练 180 轮
        batch=8,
        device=0,
        workers=2,
        amp=True,
        imgsz=640,
        lr0=0.005,
        cos_lr=True,
        patience=60,
        close_mosaic=10,
        val=True,
        plots=True,
        project=os.path.join(BASE, 'runs'),
        name='pcb_v4',        # 全新 run, 彻底避开污染
        exist_ok=True,
    )

    val = model.val(data=DATA, batch=8, device=0)
    print(f'\n训练完成! 最终结果:')
    print(f'  最终 mAP50   = {val.box.map50:.4f}')
    print(f'  最终 mAP50-95 = {val.box.map:.4f}')


if __name__ == '__main__':
    main()
