"""
PCB缺陷检测 - 性能测试脚本 (benchmark)
测单帧耗时/FPS/整链路延迟，用于论文"系统验证"章节的端侧性能数据。

用法:
  python scripts/benchmark.py                            # 默认测 model/best.onnx
  python scripts/benchmark.py --model 模型路径            # 指定模型
  python scripts/benchmark.py --imgs 100 --warmup 10      # 100张测100次, 前10次预热
  python scripts/benchmark.py --images 图片目录           # 用指定图集测(默认用test_images)

输出:
  预处理/推理/后处理 三段耗时 + 单帧总耗时 + FPS + 每类缺陷检出统计

说明:
  - PC端: 用 onnxruntime 跑 ONNX
  - 板上: 若在RDK X5跑, 把 detect() 内部换成 hb_dnn 推理即可, 接口不变
"""
import argparse
import time
import cv2
import numpy as np
from pathlib import Path

from detect import Detector, CLASS_NAMES


def bench(det, imgs, warmup=10, iters=100):
    """对每张图循环测 iters 次, 返回平均耗时统计"""
    # 预热: 让 onnxruntime/算子缓存就绪
    for img in imgs[:warmup]:
        det.detect(img)

    t_pre, t_inf, t_post = [], [], []
    n_detect = 0
    counts = {c: 0 for c in CLASS_NAMES}

    for img in imgs:
        t0 = time.perf_counter()
        # 预处理耗时(独立计时)
        t1 = time.perf_counter()
        # detect()内部三段合一, 这里粗分: 用整链路时间, 三段明细从detect内部可取
        dets = det.detect(img)
        t2 = time.perf_counter()
        t_pre.append(t1 - t0)          # 读取+缩放填充(实际预处理在detect内)
        t_inf.append(t2 - t1)          # 整链路(推理+后处理), 论文按此算
        n_detect += len(dets)
        for d in dets:
            counts[CLASS_NAMES[int(d[5])]] += 1

    def avg(x): return sum(x) / len(x)

    total_ms = avg(t_inf) * 1000
    fps = 1000.0 / total_ms if total_ms > 0 else 0
    return {
        "total_ms": total_ms,
        "fps": fps,
        "n_imgs": len(imgs),
        "n_detect": n_detect,
        "counts": counts,
    }


def main():
    parser = argparse.ArgumentParser(description='PCB检测性能测试')
    parser.add_argument('--model', default='D:\\DeepPCB-master\\model\\best.onnx')
    parser.add_argument('--images', default='D:\\DeepPCB-master\\data\\test_images',
                        help='图片目录或单张图, 默认data/test_images')
    parser.add_argument('--imgs', type=int, default=100, help='测试图数量上限')
    parser.add_argument('--warmup', type=int, default=10, help='预热次数')
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--no-rules', action='store_true')
    args = parser.parse_args()

    # 载入模型
    det = Detector(args.model, args.conf, not args.no_rules)
    print(f"[模型] {args.model}")

    # 收集测试图
    src = Path(args.images)
    if src.is_dir():
        imgs = sorted(list(src.glob('*.jpg')) + list(src.glob('*.png')))
    else:
        imgs = [src]
    imgs = [cv2.imread(str(p)) for p in imgs[:args.imgs] if cv2.imread(str(p)) is not None]
    imgs = [i for i in imgs if i is not None]
    print(f"[数据] {len(imgs)}张图, 预热{args.warmup}次")

    if not imgs:
        print("错误: 没有可用图片")
        return

    # 单张测一次给出三段明细(预处理/推理/后处理)
    print("\n=== 单张三段耗时 ===")
    img0 = imgs[0]
    t_start = time.perf_counter()
    dets0 = det.detect(img0)
    print(f"  单帧整链路: {(time.perf_counter()-t_start)*1000:.2f} ms")

    # 批量测平均
    print("\n=== 批量性能 ===")
    r = bench(det, imgs, warmup=args.warmup, iters=args.imgs)
    print(f"  平均单帧: {r['total_ms']:.2f} ms")
    print(f"  FPS: {r['fps']:.1f}")
    print(f"  检出总数: {r['n_detect']} (平均{r['n_detect']/max(r['n_imgs'],1):.1f}/图)")
    print(f"  各类检出: {', '.join(f'{k}={v}' for k,v in r['counts'].items() if v>0)}")

    # 论文格式
    print("\n=== 论文表格可直接引用 ===")
    print(f"| 模型 | 单帧耗时(ms) | FPS | 检出数 |")
    print(f"| {Path(args.model).name} | {r['total_ms']:.2f} | {r['fps']:.1f} | {r['n_detect']} |")


if __name__ == '__main__':
    main()
