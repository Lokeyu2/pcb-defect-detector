"""
重建并续训合并数据集 PCB 模型（v3）
- 起点: runs/runs/pcb_merged_v1/weights/last.pt  (epoch92, 合并数据集, PCB 6类)
- 原因: pcb_v2 权重被误训(COCO)覆盖, 这是现存最优的合并训练成果
- 全绝对路径, 在任何目录运行都安全
- 已加 __main__ 保护, 解决 Windows multiprocessing 报错
用法: python D:/DeepPCB-master/scripts/train_pcb_v3.py
"""
import os
from ultralytics import YOLO


def main():
    BASE = r'D:\DeepPCB-master'
    START = os.path.join(BASE, 'runs', 'runs', 'pcb_merged_v1', 'weights', 'last.pt')
    DATA = os.path.join(BASE, 'data', 'data.yaml')

    assert os.path.exists(START), f'起点权重不存在: {START}'
    assert os.path.exists(DATA), f'数据配置不存在: {DATA}'

    model = YOLO(START)
    print(f'从 {START} 续训 (PCB 6类, 10,342张合并数据集)')

    results = model.train(
        data=DATA,
        epochs=180,           # 从 epoch92 权重完整训练到 180
        batch=8,              # 保持 8
        device=0,             # GPU
        workers=2,
        amp=True,
        imgsz=640,
        lr0=0.005,
        cos_lr=True,
        patience=30,
        close_mosaic=10,
        val=True,
        plots=True,
        project=os.path.join(BASE, 'runs'),
        name='pcb_v3',        # 新 run, 不覆盖任何旧成果
        exist_ok=True,
    )

    # 完整验证
    val = model.val(data=DATA, batch=8, device=0)
    print(f'\n训练完成! 结果在 runs/pcb_v3/')
    print(f'最终 mAP50   = {val.box.map50:.4f}')
    print(f'最终 mAP50-95 = {val.box.map:.4f}')


if __name__ == '__main__':
    main()
