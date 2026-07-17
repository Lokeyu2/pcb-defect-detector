"""
PCB缺陷检测 - PC端推理（ONNX）
用法:
  python scripts/detect.py                          # 跑data/test_images所有图
  python scripts/detect.py --source 图片路径          # 单张图
  python scripts/detect.py --source 0                # 摄像头
  python scripts/detect.py --conf 0.5                # 统一阈值
  python scripts/detect.py --no-rules                # 不加后处理过滤
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
import onnxruntime as rt

CLASS_NAMES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]
CLASS_COLORS = [(0,0,255),(255,0,0),(0,255,0),(255,255,0),(255,0,255),(0,255,255)]


class Detector:
    def __init__(self, onnx_path, conf=0.5, use_rules=True):
        self.conf = conf
        self.use_rules = use_rules
        self.session = rt.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, img):
        """输入BGR图，返回 [[x1,y1,x2,y2,conf,cls_id], ...]"""
        h, w = img.shape[:2]
        # 缩放填充到640x640
        scale = min(640/w, 640/h)
        nw, nh = int(w*scale), int(h*scale)
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = resized
        # 转为模型输入
        tensor = canvas.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
        # 推理
        out = self.session.run(None, {self.input_name: tensor})[0][0]
        cx, cy, bw, bh = out[:4]
        scores = out[4:]  # [6, 8400]

        # 解析检测结果
        results = []
        for i in range(scores.shape[1]):
            s = scores[:, i]
            max_s, cls_id = float(s.max()), int(s.argmax())
            if max_s < self.conf:
                continue
            x1 = int((cx[i] - bw[i]/2) / scale)
            y1 = int((cy[i] - bh[i]/2) / scale)
            x2 = int((cx[i] + bw[i]/2) / scale)
            y2 = int((cy[i] + bh[i]/2) / scale)
            results.append([x1, y1, x2, y2, max_s, cls_id])

        if not results:
            return np.empty((0, 6))

        # NMS（各类别分开做）
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

        if not final:
            return np.empty((0, 6))

        final = np.array(final)

        # 后处理规则
        if self.use_rules:
            keep = []
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for d in final:
                x1, y1, x2, y2, sc, ci = d.astype(int)
                name = CLASS_NAMES[ci]
                wb, hb = x2-x1, y2-y1
                # 规则1: 极小面积剔除
                if wb * hb < 5:
                    continue
                # 规则2: 圆形框+亮中心 → 过孔 → 剔除open
                if name == "open" and wb >= 6 and hb >= 6:
                    aspect = min(wb/hb, hb/wb)
                    roi = gray[y1:y2, x1:x2]
                    if roi.size > 0 and aspect > 0.7:
                        cx, cy = wb//2, hb//2
                        center = roi[max(0,cy-2):cy+3, max(0,cx-2):cx+3]
                        if center.size > 0 and (center > 200).mean() > 0.3:
                            continue
                # 规则3: 拐角/T型 → 剔除short/spur
                if name in ("short", "spur"):
                    roi = gray[y1:y2, x1:x2]
                    if roi.size >= 25:
                        edges = cv2.Canny(roi, 30, 100)
                        if edges.sum(axis=0).var() > 10 and edges.sum(axis=1).var() > 10:
                            continue
                keep.append(d)
            final = np.array(keep) if keep else np.empty((0, 6))

        return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='data/test_images', help='图片/目录/摄像头ID')
    parser.add_argument('--model', default='model/best.onnx')
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--save-dir', default='results')
    parser.add_argument('--no-rules', action='store_true')
    args = parser.parse_args()

    det = Detector(args.model, args.conf, not args.no_rules)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # 获取输入
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
        while True:
            ret, frame = cap.read()
            if not ret: break
            dets = det.detect(frame)
            for d in dets:
                x1, y1, x2, y2, sc, ci = d.astype(int)
                cv2.rectangle(frame, (x1,y1), (x2,y2), CLASS_COLORS[ci], 2)
                label = f'{CLASS_NAMES[ci]} {sc:.2f}'
                (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1,y1-th-6), (x1+tw+4,y1), CLASS_COLORS[ci], -1)
                cv2.putText(frame, label, (x1+2,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.imshow('PCB Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()
        return

    # 图片模式
    src = Path(args.source)
    imgs = sorted(src.glob('*.jpg') + src.glob('*.png')) if src.is_dir() else [src]
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None: continue
        dets = det.detect(img)
        for d in dets:
            x1, y1, x2, y2, sc, ci = d.astype(int)
            cv2.rectangle(img, (x1,y1), (x2,y2), CLASS_COLORS[ci], 2)
            label = f'{CLASS_NAMES[ci]} {sc:.2f}'
            (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1,y1-th-6), (x1+tw+4,y1), CLASS_COLORS[ci], -1)
            cv2.putText(img, label, (x1+2,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        out = save_dir / f'result_{p.name}'
        cv2.imwrite(str(out), img)
        counts = {}
        for d in dets:
            n = CLASS_NAMES[int(d[5])]
            counts[n] = counts.get(n, 0) + 1
        detail = ', '.join(f'{k}={v}' for k,v in sorted(counts.items()))
        print(f'[{p.name}] {len(dets)} defects | {detail}')
    print(f'\n完成: {len(imgs)}张, 结果存于 {save_dir}/')


if __name__ == '__main__':
    main()
