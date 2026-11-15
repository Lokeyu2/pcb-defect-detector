"""
FP32 ONNX 精度评估 (Windows, onnxruntime)
用法: python eval_fp32.py
- 模型: model/best.onnx (opset11, pcb_v4)
- 与 eval_int8.py 相同的检测/NMS/匹配逻辑, 保证对比公平
"""
import os
import sys
import numpy as np
import cv2
import onnxruntime as rt
from pathlib import Path

import paths

VAL_IMG = paths.VAL_IMG
VAL_LAB = paths.VAL_LAB
FP32_MODEL = str(paths.ONNX_PATH)
CLASS_NAMES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]
CONF = 0.5


def load_labels(p):
    out = []
    if Path(p).exists():
        for line in open(p):
            q = line.strip().split()
            if len(q) >= 5:
                out.append([int(q[0])] + list(map(float, q[1:])))
    return out


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    aa = (a[2]-a[0])*(a[3]-a[1]); bb = (b[2]-b[0])*(b[3]-b[1])
    return inter/(aa+bb-inter) if aa+bb-inter > 0 else 0


def detect(sess, img):
    """与 detect.py 一致: letterbox 640x640 -> /255 -> fp32 -> 逐类NMS"""
    h, w = img.shape[:2]
    scale = min(640/w, 640/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    tensor = canvas.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
    out = sess.run(None, {'images': tensor})[0][0]  # [10, 8400]
    cx, cy, bw, bh = out[:4]
    scores = out[4:]

    results = []
    for i in range(scores.shape[1]):
        s = scores[:, i]
        mx, ci = float(s.max()), int(s.argmax())
        if mx < CONF:
            continue
        x1 = int((cx[i] - bw[i]/2) / scale)
        y1 = int((cy[i] - bh[i]/2) / scale)
        x2 = int((cx[i] + bw[i]/2) / scale)
        y2 = int((cy[i] + bh[i]/2) / scale)
        results.append([x1, y1, x2, y2, mx, ci])

    if not results:
        return np.empty((0, 6))
    boxes = np.array(results)
    final = []
    for ci in range(6):
        mask = boxes[:, 5] == ci
        if not mask.any():
            continue
        sub = boxes[mask]
        keep = cv2.dnn.NMSBoxes(sub[:, :4].tolist(), sub[:, 4].tolist(), 0.01, 0.45)
        if keep is not None:
            final.extend(sub[keep.flatten()].tolist())
    return np.array(final) if final else np.empty((0, 6))


def main():
    paths.require_dataset()
    sess = rt.InferenceSession(FP32_MODEL)
    imgs = sorted([f for f in os.listdir(VAL_IMG) if f.endswith('.jpg')])
    total_gt = {i: 0 for i in range(6)}
    total_tp = {i: 0 for i in range(6)}
    total_fp = {i: 0 for i in range(6)}
    total_fn = {i: 0 for i in range(6)}
    n_img = 0

    for fname in imgs:
        ip = VAL_IMG / fname
        labels = load_labels(VAL_LAB / (fname.replace('.jpg', '.txt')))
        if not labels:
            continue
        with open(str(ip), 'rb') as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            continue
        n_img += 1
        ih, iw = img.shape[:2]
        gts = []
        for l in labels:
            x, y, w, h = l[1], l[2], l[3], l[4]
            gts.append([l[0], int((x-w/2)*iw), int((y-h/2)*ih), int((x+w/2)*iw), int((y+h/2)*ih)])
        for g in gts:
            total_gt[g[0]] += 1

        dets = detect(sess, img)
        matched = set()
        for d in dets:
            best_iou, best_gt = 0.3, -1
            for gi, g in enumerate(gts):
                if gi in matched:
                    continue
                v = iou(d[:4], g[1:])
                if v > best_iou:
                    best_iou, best_gt = v, gi
            if best_gt >= 0 and gts[best_gt][0] == int(d[5]):
                total_tp[int(d[5])] += 1
                matched.add(best_gt)
            else:
                total_fp[int(d[5])] += 1
        for gi, g in enumerate(gts):
            if gi not in matched:
                total_fn[g[0]] += 1

    print(f'=== FP32 模型评估 (pcb_v4, conf={CONF}, {n_img}张含GT图) ===')
    print(f'{"类别":<12}{"GT":>6}{"TP":>5}{"FP":>5}{"FN":>5}{"P":>8}{"R":>8}{"F1":>8}')
    for c in range(6):
        p = total_tp[c]/(total_tp[c]+total_fp[c]) if total_tp[c]+total_fp[c] > 0 else 0
        r = total_tp[c]/(total_tp[c]+total_fn[c]) if total_tp[c]+total_fn[c] > 0 else 0
        f1 = 2*p*r/(p+r) if p+r > 0 else 0
        gt = total_tp[c]+total_fn[c]
        print(f'{CLASS_NAMES[c]:<12}{gt:>6}{total_tp[c]:>5}{total_fp[c]:>5}{total_fn[c]:>5}{p:>8.3f}{r:>8.3f}{f1:>8.3f}')
    p_all = sum(total_tp.values())/(sum(total_tp.values())+sum(total_fp.values())) if (sum(total_tp.values())+sum(total_fp.values())) > 0 else 0
    r_all = sum(total_tp.values())/(sum(total_tp.values())+sum(total_fn.values())) if (sum(total_tp.values())+sum(total_fn.values())) > 0 else 0
    f1_all = 2*p_all*r_all/(p_all+r_all) if p_all+r_all > 0 else 0
    print(f'{"总计":<12}{sum(total_gt.values()):>6}{sum(total_tp.values()):>5}{sum(total_fp.values()):>5}{sum(total_fn.values()):>5}{p_all:>8.3f}{r_all:>8.3f}{f1_all:>8.3f}')


if __name__ == '__main__':
    main()
