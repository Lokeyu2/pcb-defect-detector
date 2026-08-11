"""
验证集混淆矩阵分析 - 定位实拍图上类别系统性判错的问题
跑 pcb_v4 模型, 对照 GT, 输出各类别的 精确率/召回率 + 类别间混淆计数
"""
import cv2, numpy as np, sys, collections, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from detect import Detector

VAL_IMG = Path('D:/DeepPCB_YOLO/images/val')
VAL_LAB = Path('D:/DeepPCB_YOLO/labels/val')
ONNX = r'D:\DeepPCB-master\model\best.onnx'   # 修复后的 pcb_v4
CLASS_NAMES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]
CONF = 0.5

def load_labels(p):
    out = []
    if p.exists():
        for line in open(p):
            q = line.strip().split()
            if len(q) >= 5:
                out.append([int(q[0])] + list(map(float, q[1:])))
    return out

def iou(a, b):
    x1 = max(a[0],b[0]); y1 = max(a[1],b[1])
    x2 = min(a[2],b[2]); y2 = min(a[3],b[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    aa = (a[2]-a[0])*(a[3]-a[1]); bb = (b[2]-b[0])*(b[3]-b[1])
    return inter/(aa+bb-inter) if aa+bb-inter>0 else 0

det = Detector(ONNX, conf=CONF, use_rules=False)

conf_mat = np.zeros((6,6), dtype=int)
tp = np.zeros(6); fp = np.zeros(6); fn_arr = np.zeros(6)
img_stats = collections.Counter()

imgs = sorted([f for f in os.listdir(VAL_IMG) if f.endswith('.jpg')])  # Path.glob在此环境失效, 用os.listdir
n = 0
for fname in imgs:
    ip = VAL_IMG / fname
    labels = load_labels(VAL_LAB / (fname.replace('.jpg', '.txt')))
    if not labels: continue
    with open(str(ip), 'rb') as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None: continue
    raw = det.detect(img)
    img_stats[len(raw)] += 1
    n += 1

    ih, iw = img.shape[:2]
    gts = []
    for l in labels:
        x,y,w,h = l[1],l[2],l[3],l[4]
        gts.append([l[0], int((x-w/2)*iw), int((y-h/2)*ih), int((x+w/2)*iw), int((y+h/2)*ih)])
    gt_ids = set()
    for d in raw:
        best_iou, best_gt = 0.3, -1
        for gi, g in enumerate(gts):
            if gi in gt_ids: continue
            v = iou(d[:4].astype(int), g[1:])
            if v > best_iou: best_iou, best_gt = v, gi
        if best_gt >= 0:
            gt_cls = gts[best_gt][0]; pred_cls = int(d[5])
            conf_mat[gt_cls][pred_cls] += 1
            tp[gt_cls] += 1
            gt_ids.add(best_gt)
        else:
            fp[int(d[5])] += 1
    for gi, g in enumerate(gts):
        if gi not in gt_ids:
            fn_arr[g[0]] += 1

print(f'=== 验证集混淆分析 (pcb_v4, conf={CONF}, {n}张含GT图) ===')
print(f'\n--- 类别级指标 ---')
print(f'{"类别":<12}{"GT":>6}{"TP":>5}{"FP":>5}{"FN":>5}{"P":>8}{"R":>8}{"F1":>8}')
for c in range(6):
    p = tp[c]/(tp[c]+fp[c]) if tp[c]+fp[c]>0 else 0
    r = tp[c]/(tp[c]+fn_arr[c]) if tp[c]+fn_arr[c]>0 else 0
    f1 = 2*p*r/(p+r) if p+r>0 else 0
    gt = tp[c]+fn_arr[c]
    print(f'{CLASS_NAMES[c]:<12}{gt:>6}{tp[c]:>5}{fp[c]:>5}{fn_arr[c]:>5}{p:>8.3f}{r:>8.3f}{f1:>8.3f}')

print(f'\n--- 混淆矩阵 (行=GT真实类, 列=预测类) ---')
print('        ' + ''.join(f'{x:>10}' for x in CLASS_NAMES))
for i in range(6):
    print(f'{CLASS_NAMES[i]:<8}' + ''.join(f'{conf_mat[i][j]:>10}' for j in range(6)))

print(f'\n--- 最大跨类混淆 (GT vs 预测) ---')
offdiag = []
for i in range(6):
    for j in range(6):
        if i != j and conf_mat[i][j] > 0:
            offdiag.append((conf_mat[i][j], CLASS_NAMES[i], CLASS_NAMES[j]))
for cnt, g, p in sorted(offdiag, reverse=True)[:8]:
    print(f'  {g} → 判成 {p}: {cnt}')

print(f'\n--- 每图检测数分布 (评估过度检测) ---')
for k in sorted(img_stats):
    print(f'  {k}个缺陷: {img_stats[k]}张')
