"""
正确续训 pcb_v3 合并数据集 PCB 模型
- 从 runs/pcb_v3/weights/last.pt 真正续训 (resume=True)
- 修正上次 resume=False 导致 early-stop 提前的问题
- patience 给足, 让模型充分收敛到最优
用法: python D:/DeepPCB-master/scripts/train_pcb_v3_resume.py
"""
import os
from ultralytics import YOLO


def main():
    BASE = r'D:\DeepPCB-master'
    RUN_DIR = os.path.join(BASE, 'runs', 'pcb_v3')
    LAST = os.path.join(RUN_DIR, 'weights', 'last.pt')
    DATA = os.path.join(BASE, 'data', 'data.yaml')

    assert os.path.exists(LAST), f'续训起点不存在: {LAST}'
    assert os.path.exists(DATA), f'数据配置不存在: {DATA}'

    model = YOLO(LAST)
    print(f'从 {LAST} 续训 (正确累计epoch, patience=60)')

    results = model.train(
        resume=True,          # 关键: 累计epoch续训, 不再当全新训练
        epochs=220,           # 续训到 220 轮(累计)
        batch=8,              # 保持 8
        device=0,             # GPU
        workers=2,
        amp=True,
        imgsz=640,
        patience=60,          # 给足耐心, 避免误早停
        close_mosaic=10,
        val=True,
        plots=True,
        project=os.path.join(BASE, 'runs'),
        name='pcb_v3',
        exist_ok=True,
    )

    val = model.val(data=DATA, batch=8, device=0)
    print(f'\n训练完成! 最终结果:')
    print(f'  最终 mAP50   = {val.box.map50:.4f}')
    print(f'  最终 mAP50-95 = {val.box.map:.4f}')


if __name__ == '__main__':
    main()
