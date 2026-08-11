"""
后处理规则有效性验证脚本
- 用同一批检测结果, 对比 无规则 / 现有规则 / 现有+新增规则 的 Prec/Recall
- 验证集: D:/DeepPCB_YOLO/images/val + labels/val (1826张)
- 输出对比表, 供论文和规则调优使用
用法: python scripts/eval_rules.py
"""
import cv2
import numpy as np
import onnxruntime as rt
from pathlib import Path

CLASS_NAMES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]

# ---------- 检测器（复用 detect.py 的 Detector, 支持开关规则） ----------
import sys
sys.path.insert(0, str(Path(__file__).parent))
from detect import Detector

# ---------- 评估器 ----------
def load_labels(txt_path):
    labels = []
    if txt_path.exists():
        for line in open(txt_path):
            p = line.strip().split()
            if len(p) >= 5:
                labels.append([int(p[0])] + list(map(float, p[1:])))
    return labels

def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0

def evaluate(det, img_dir, label_dir, img_ext='.jpg'):
    """返回 (total_gt, total_det, total_tp, total_fp, total_fn) 全类合计"""
    total_gt = total_det = total_tp = total_fp = total_fn = 0
    imgs = sorted(img_dir.glob('*' + img_ext)) if img_ext else sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
    for img_path in imgs:
        label_path = label_dir / (img_path.stem + '.txt')
        gts = load_labels(label_path)
        if not gts: continue
        img = cv2.imread(str(img_path))
        if img is None: continue
        h_img, w_img = img.shape[:2]
        dets = det.detect(img)
        total_gt += len(gts)
        total_det += len(dets)
        matched = set()
        for d in dets:
            best_iou, best_gt = 0.5, -1
            for gi, gt in enumerate(gts):
                if gi in matched or d[5] != gt[0]: continue
                gt_box = [int((gt[1]-gt[3]/2)*w_img), int((gt[2]-gt[4]/2)*h_img),
                          int((gt[1]+gt[3]/2)*w_img), int((gt[2]+gt[4]/2)*h_img)]
                v = iou(d[:4], gt_box)
                if v > best_iou:
                    best_iou, best_gt = v, gi
            if best_gt >= 0:
                total_tp += 1; matched.add(best_gt)
            else:
                total_fp += 1
        total_fn += len(gts) - len(matched)
    return total_gt, total_det, total_tp, total_fp, total_fn

def report(tag, gt, det, tp, fp, fn):
    prec = tp/(tp+fp) if tp+fp>0 else 0
    rec = tp/(tp+fn) if tp+fn>0 else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec>0 else 0
    print(f'{tag:28s} GT={gt:5d} 检测={det:5d}  TP={tp:5d} FP={fp:5d} FN={fn:5d}  '
          f'Prec={prec:.3f} Recall={rec:.3f} F1={f1:.3f}')
    return prec, rec, f1


if __name__ == '__main__':
    # 用验证集评估（有GT标签）
    val_img = Path('D:/DeepPCB_YOLO/images/val')
    val_lab = Path('D:/DeepPCB_YOLO/labels/val')
    # 用 test_images 统计检测数量（无GT, 看规则过滤效果）
    test_img = Path('data/test_images')

    print('='*85)
    print('后处理规则有效性验证')
    print('验证集: D:/DeepPCB_YOLO/images/val (1826张, 有GT标签)')
    print('='*85)

    onnx_path = 'model/best.onnx'
    results = {}

    for tag, use_rules in [('无规则 (baseline)', False), ('现有9条规则', True)]:
        det = Detector(onnx_path, conf=0.5, use_rules=use_rules)
        gt, det_n, tp, fp, fn = evaluate(det, val_img, val_lab)
        print(f'\n--- {tag} ---')
        p, r, f1 = report(tag, gt, det_n, tp, fp, fn)
        results[tag] = (p, r, f1, fp, fn)

    print('\n' + '='*85)
    print('对比结论 (有GT验证集):')
    base = results['无规则 (baseline)']
    cur = results['现有9条规则']
    print(f'  无规则:  Prec={base[0]:.3f} Recall={base[1]:.3f} F1={base[2]:.3f}')
    print(f'  现有9条: Prec={cur[0]:.3f} Recall={cur[1]:.3f} F1={cur[2]:.3f}')
    print(f'  → 现有规则把 Prec 提升 {(cur[0]-base[0])*100:.1f}pp, Recall 变化 {(cur[1]-base[1])*100:.1f}pp')
    print('='*85)
