"""
生成地平线工具链校准集（calibration_bin）
从训练集抽200张图, 按 detect.py 预处理生成 float32 NCHW 的 .bin 校准文件

注意: detect.py 用 BGR 读图做 letterbox+归一化; 地平线配置 input_type_rt 是 rgb
      -> 这里必须显式 BGR->RGB, 保证颜色通道顺序与模型训练一致
"""
import os
import random
import numpy as np
import cv2
from pathlib import Path

import paths

TRAIN_DIR = paths.TRAIN_IMG
OUT_DIR = paths.PROJECT_ROOT / 'calibration_bin'
NUM_CALIB = 200          # 地平线官方建议 100~200 张
SIZE = 640

random.seed(42)          # 可复现
np.random.seed(42)


def letterbox(img, new_shape=(SIZE, SIZE), color=114):
    """与 detect.py 一致: 等比例缩放+灰边填充"""
    h, w = img.shape[:2]
    scale = min(new_shape[0] / w, new_shape[1] / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new_shape[1], new_shape[0], 3), color, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas


def main():
    imgs = sorted([f for f in os.listdir(TRAIN_DIR) if f.endswith('.jpg')])
    if len(imgs) < NUM_CALIB:
        print(f'警告: 训练集只有 {len(imgs)} 张, 少于校准需求 {NUM_CALIB}')
    chosen = random.sample(imgs, min(NUM_CALIB, len(imgs)))

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    for i, fname in enumerate(chosen):
        p = TRAIN_DIR / fname
        img = paths.imread(p)                         # BGR（支持中文路径）
        if img is None:
            print(f'跳过无法读取: {fname}')
            continue
        canvas = letterbox(img)                        # 640x640 BGR
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)  # -> RGB
        tensor = rgb.astype(np.float32).transpose(2, 0, 1)[None]  # [1,3,640,640] float32
        tensor.tofile(str(OUT_DIR / f'calib_{i:04d}.rgb'))
    print(f'完成: 生成 {min(NUM_CALIB, len(chosen))} 个校准文件 -> {OUT_DIR}/')


if __name__ == '__main__':
    main()
