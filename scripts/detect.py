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

        # 后处理规则（针对真实PCB板摄像头实拍优化）
        if self.use_rules:
            keep = []
            h_img, w_img = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 基于原图统计自适应阈值: 明亮区域通常是铜箔/焊盘, 暗区是基板
            bg_bright = np.percentile(gray, 85)  # 背景高亮阈值（铜箔面）

            for d in final:
                x1, y1, x2, y2 = d[:4].astype(int)   # 坐标取整
                sc = float(d[4])                      # 置信度必须保留小数!
                ci = int(d[5])
                name = CLASS_NAMES[ci]
                wb, hb = x2-x1, y2-y1
                area = wb * hb
                aspect = min(wb/hb, hb/wb) if wb > 0 and hb > 0 else 0

                # ---- 通用过滤规则 (不分类别) ----

                # R1: 极小面积剔除 (真实缺陷至少要有一定尺寸)
                if area < 8:
                    continue

                # R2: 极端宽高比 → 大概率是走线/划痕误检
                if aspect < 0.15:
                    continue

                # R3: 框贴近图像边缘 (10px以内) 且置信度 < 0.6 → 边缘伪影剔除
                edge_dist = min(x1, y1, w_img - x2, h_img - y2)
                if edge_dist < 10 and sc < 0.6:
                    continue

                # R4: 框完全位于大面积纯色区域 → 基板纹理误检
                roi = gray[y1:y2, x1:x2]
                if roi.size > 0 and roi.std() < 8 and sc < 0.6:
                    continue

                # ---- 类别特定过滤 ----

                # R5: open (开路) → 过孔/焊盘误检剔除
                if name == "open":
                    # 5a: 宽高比 > 0.6 说明是圆形/方形 → 可能是过孔或焊盘
                    if aspect > 0.6:
                        # 检查中心区域是否为高亮(铜箔/焊盘特征)
                        cx_roi, cy_roi = wb//2, hb//2
                        center = roi[max(0,cy_roi-3):cy_roi+4, max(0,cx_roi-3):cx_roi+4]
                        if center.size > 0:
                            center_mean = center.mean()
                            # 焊盘/铜箔中心高亮 或者 过孔中心暗(孔洞)
                            if center_mean > bg_bright * 0.9 or center_mean < 40:
                                if sc < 0.65:
                                    continue
                    # 5b: 面积中等且置信度低 → 可能是焊盘
                    if 20 < area < 200 and sc < 0.45:
                        continue

                # R6: short (短路) / spur (毛刺) → 走线/焊盘误检剔除
                if name in ("short", "spur"):
                    # 6a: ROI内边缘复杂度高且置信度低 → 走线密集区误检
                    if roi.size >= 36:
                        edges = cv2.Canny(roi, 30, 100)
                        edge_ratio = (edges > 0).mean()
                        # 走线密集区边缘占比高(>30%)且杂乱
                        if edge_ratio > 0.3 and sc < 0.55:
                            continue
                    # 6b: 极小框+低置信度 → 噪声
                    if area < 15 and sc < 0.5:
                        continue

                # R7: copper (铜箔/多余铜) → 背景纹理/焊盘误检剔除
                if name == "copper":
                    # 7a: 框内亮度均匀且置信度低 → 基板纹理
                    if roi.size > 0 and roi.std() < 12 and sc < 0.55:
                        continue
                    # 7b: 小面积+低置信度
                    if area < 25 and sc < 0.5:
                        continue

                # R8: pin-hole (针孔) → 丝印字符/污渍误检剔除
                if name == "pin-hole":
                    # 8a: 低置信度直接剔除(针孔太小, 低置信几乎都是误检)
                    if sc < 0.40:
                        continue
                    # 8b: 检查邻域是否有丝印/字符特征
                    if wb >= 4 and hb >= 4 and sc < 0.70:
                        x1e, y1e = max(0, x1-10), max(0, y1-10)
                        x2e, y2e = min(gray.shape[1], x2+10), min(gray.shape[0], y2+10)
                        surround = gray[y1e:y2e, x1e:x2e].copy()
                        mask = np.ones_like(surround, dtype=np.uint8)
                        mask[y1-y1e:y1-y1e+hb, x1-x1e:x1-x1e+wb] = 0
                        bg = surround[mask.astype(bool)]
                        if bg.size > 0:
                            bg_mean, bg_std = bg.mean(), bg.std()
                            # 丝印文字特征: 中高亮度(白色文字) + 高方差(笔画与背景交错)
                            if 70 < bg_mean < 190 and bg_std > 35:
                                continue

                # R9: mousebite (鼠咬) → 走线拐角误检剔除
                if name == "mousebite":
                    if area < 12 and sc < 0.5:
                        continue
                    # 鼠咬通常呈半圆形凹陷, ROI内边缘分布不均匀
                    if roi.size >= 36 and sc < 0.55:
                        edges = cv2.Canny(roi, 30, 100)
                        h_edge = (edges.sum(axis=0) > 0).sum()
                        v_edge = (edges.sum(axis=1) > 0).sum()
                        # 如果水平和垂直边缘占比都很高 → 走线交叉/拐角误检
                        if h_edge > wb * 0.5 and v_edge > hb * 0.5:
                            continue

                keep.append(d)
            final = np.array(keep) if keep else np.empty((0, 6))

        return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='data/test_images', help='图片/目录/摄像头ID')
    parser.add_argument('--model', default='D:\\DeepPCB-master\\model\\best.onnx')
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
                x1, y1, x2, y2 = d[:4].astype(int)
                sc, ci = float(d[4]), int(d[5])
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
    imgs = sorted(list(src.glob('*.jpg')) + list(src.glob('*.png'))) if src.is_dir() else [src]
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None: continue
        dets = det.detect(img)
        for d in dets:
            x1, y1, x2, y2 = d[:4].astype(int)
            sc, ci = float(d[4]), int(d[5])
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
