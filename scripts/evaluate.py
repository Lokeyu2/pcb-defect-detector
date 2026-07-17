"""
批量验证 + 对比表格生成
用法: python scripts/evaluate.py
"""
import cv2
import numpy as np
import onnxruntime as rt
from pathlib import Path

CLASS_NAMES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]

def load_labels(txt_path):
    """读取YOLO格式标签"""
    labels = []
    if not txt_path.exists():
        return labels
    for line in open(txt_path):
        parts = line.strip().split()
        if len(parts) >= 5:
            labels.append([int(parts[0])] + list(map(float, parts[1:])))
    return labels

def detect(session, img, conf=0.5):
    h, w = img.shape[:2]
    scale = min(640/w, 640/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((640,640,3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    tensor = canvas.astype(np.float32).transpose(2,0,1)[np.newaxis] / 255.0
    out = session.run(None, {'images': tensor})[0][0]
    cx, cy, bw, bh = out[:4]
    scores = out[4:]

    results = []
    for i in range(scores.shape[1]):
        s = scores[:, i]
        mx, ci = float(s.max()), int(s.argmax())
        if mx < conf: continue
        x1 = int((cx[i]-bw[i]/2)/scale)
        y1 = int((cy[i]-bh[i]/2)/scale)
        x2 = int((cx[i]+bw[i]/2)/scale)
        y2 = int((cy[i]+bh[i]/2)/scale)
        results.append([x1, y1, x2, y2, mx, ci])

    # NMS
    final = []
    if results:
        boxes = np.array(results)
        for ci in range(6):
            mask = boxes[:,5]==ci
            if not mask.any(): continue
            sub = boxes[mask]
            keep = cv2.dnn.NMSBoxes(sub[:,:4].tolist(), sub[:,4].tolist(), 0.01, 0.45)
            if keep is not None:
                final.extend(sub[keep.flatten()].tolist())
    return final

def iou(b1, b2):
    """计算两个框的IoU"""
    x1, y1, x2, y2 = [max(b1[0], b2[0]), max(b1[1], b2[1]),
                       min(b1[2], b2[2]), min(b1[3], b2[3])]
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1 + a2 - inter) if (a1+a2-inter) > 0 else 0


if __name__ == '__main__':
    session = rt.InferenceSession('model/best.onnx')
    img_dir = Path('data/test_images')
    label_dir = Path('D:/DeepPCB_YOLO/labels/val')
    conf = 0.5

    total_gt = {i:0 for i in range(6)}
    total_tp = {i:0 for i in range(6)}
    total_fp = {i:0 for i in range(6)}
    total_fn = {i:0 for i in range(6)}
    image_count = 0

    for img_path in sorted(img_dir.glob('*.jpg')):
        label_path = label_dir / img_path.name.replace('.jpg', '.txt')
        gts = load_labels(label_path)
        if not gts:
            continue

        image_count += 1
        img = cv2.imread(str(img_path))
        dets = detect(session, img, conf)

        # 统计GT
        for gt in gts:
            total_gt[gt[0]] += 1

        # 匹配检测框
        matched_gt = set()
        for d in dets:
            best_iou = 0.5  # IoU阈值
            best_gt = -1
            for gi, gt in enumerate(gts):
                if gi in matched_gt: continue
                if d[5] != gt[0]: continue  # 类别不同跳过
                # 转像素坐标
                gt_box = [
                    int((gt[1]-gt[3]/2)*640), int((gt[2]-gt[4]/2)*640),
                    int((gt[1]+gt[3]/2)*640), int((gt[2]+gt[4]/2)*640)
                ]
                if iou(d[:4], gt_box) > best_iou:
                    best_iou = iou(d[:4], gt_box)
                    best_gt = gi
            if best_gt >= 0:
                total_tp[d[5]] += 1
                matched_gt.add(best_gt)
            else:
                total_fp[d[5]] += 1

        # 漏检
        for gi, gt in enumerate(gts):
            if gi not in matched_gt:
                total_fn[gt[0]] += 1

    # 输出报告
    print('='*65)
    print('PCB缺陷检测 - 验证集评估报告')
    print(f'测试图片: {image_count}张  置信度阈值: {conf}')
    print('='*65)
    print(f'{"类别":12s} {"GT数":6s} {"检出":6s} {"TP":6s} {"FP":6s} {"FN":6s} {"Prec":7s} {"Recall":7s}')
    print('-'*65)
    total_tp_all = sum(total_tp.values())
    total_fp_all = sum(total_fp.values())
    total_fn_all = sum(total_fn.values())
    for ci in range(6):
        tp = total_tp[ci]
        fp = total_fp[ci]
        fn = total_fn[ci]
        gt = total_gt[ci]
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        print(f'{CLASS_NAMES[ci]:12s} {gt:<6d} {tp+fp:<6d} {tp:<6d} {fp:<6d} {fn:<6d} {prec:.3f}  {rec:.3f}')
    print('-'*65)
    prec_all = total_tp_all/(total_tp_all+total_fp_all) if (total_tp_all+total_fp_all)>0 else 0
    rec_all = total_tp_all/(total_tp_all+total_fn_all) if (total_tp_all+total_fn_all)>0 else 0
    print(f'{"总计":12s} {sum(total_gt.values()):<6d} {total_tp_all+total_fp_all:<6d} {total_tp_all:<6d} {total_fp_all:<6d} {total_fn_all:<6d} {prec_all:.3f}  {rec_all:.3f}')
    print('='*65)
    print(f'mAP50 (训练记录): 0.991')
    print(f'结果已保存至 results/evaluation_report.txt')

    # 保存报告
    with open('results/evaluation_report.txt', 'w') as f:
        f.write(f'PCB缺陷检测 - 验证集评估报告\n')
        f.write(f'测试图片: {image_count}张  置信度阈值: {conf}\n\n')
        f.write(f'{"Class":12s} {"GT":6s} {"Det":6s} {"TP":6s} {"FP":6s} {"FN":6s} {"Prec":7s} {"Recall":7s}\n')
        f.write('-'*55+'\n')
        for ci in range(6):
            tp, fp, fn = total_tp[ci], total_fp[ci], total_fn[ci]
            prec = tp/(tp+fp) if (tp+fp)>0 else 0
            rec = tp/(tp+fn) if (tp+fn)>0 else 0
            f.write(f'{CLASS_NAMES[ci]:12s} {total_gt[ci]:<6d} {tp+fp:<6d} {tp:<6d} {fp:<6d} {fn:<6d} {prec:.3f}  {rec:.3f}\n')
        f.write('-'*55+'\n')
        f.write(f'{"Total":12s} {sum(total_gt.values()):<6d} {total_tp_all+total_fp_all:<6d} {total_tp_all:<6d} {total_fp_all:<6d} {total_fn_all:<6d} {prec_all:.3f}  {rec_all:.3f}\n')
