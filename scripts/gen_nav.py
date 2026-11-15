"""生成 PCB 项目文件导航 HTML（本地浏览器打开，点击直达文件）"""
import io
from pathlib import Path

R = Path(r'D:\毕业设计-代码与资料')
M = R / 'DeepPCB-master'
DS = R / 'DeepPCB_YOLO'


def fsize(p):
    p = Path(p)
    if p.is_file():
        s = p.stat().st_size
        return f"{s/1024/1024:.1f} MB" if s > 1024 * 1024 else f"{s/1024:.0f} KB"
    return "—"


def furl(p):
    return "file:///" + str(Path(p)).replace("\\", "/")


def rows_table(rows, thumb=False, extra_thumb_col=False):
    out = []
    for name, desc, size, p in rows:
        ext = Path(name).suffix.lower()
        t = ''
        if thumb and ext in ('.png', '.jpg', '.jpeg'):
            t = '<span class="tb"><img class="thumb" src="%s" loading="lazy"></span>' % furl(p)
        td_thumb = '<td class="tc">%s</td>' % t if extra_thumb_col else ''
        out.append(
            '<tr>%s<td><a href="%s">%s</a></td><td>%s</td><td class="sz">%s</td></tr>'
            % (td_thumb, furl(p), name, desc, size)
        )
    return "\n".join(out)


def pick(base, items):
    rows = []
    for name, desc in items:
        p = Path(base) / name
        if p.exists():
            rows.append((name, desc, fsize(p), p))
    return rows


docs = pick(M / 'results', [
    ('数据对比总表.html', '⭐ 全部实验数据对比（浏览器直接看 / 可打印）'),
    ('数据对比总表.md', '同上，Markdown 源文件'),
    ('量化对比报告_FP32_vs_INT8.md', '量化前后精度对比 + 根因分析'),
    ('rule_summary_20261115.txt', '后处理规则验证汇总（今天实测）'),
    ('rule_per_class_20261115.log', '规则按类别生效明细'),
    ('eval_rules_20261115_iou03.log', '规则完整跑分（IoU=0.3 口径）'),
])
docs.append(('实验日志.md', '⭐ 完整实验日志 21KB', fsize(M / '实验日志.md'), M / '实验日志.md'))

demos = pick(M / 'results' / 'demo', [
    ('demo_01_原图vs检测结果.jpg', '⭐ 左原图 / 右检测结果'),
    ('demo_02_规则过滤对比.jpg', '⭐ 左无规则 / 右有规则（贴边误检被 R3 滤掉）'),
    ('demo_03_多缺陷样例.jpg', '⭐ 一张图集齐 6 类缺陷'),
])

models = pick(M / 'model', [
    ('best.pt', '⭐ pcb_v4 训练权重（PyTorch，主力）'),
    ('best.onnx', '⭐ ONNX 导出（PC 推理 / 转 BIN）'),
    ('pcb_v4_noflip.onnx', '消融实验版（mAP50-95=0.7397）'),
    ('best_v2_old.pt', '⚠️ 旧 v2 模型，勿用'),
])

bins = pick(M / 'config' / 'pcb_v4_bin', [
    ('pcb_v4.bin', '⭐ 上板模型（RDK X5 BPU）'),
    ('pcb_v4_quantized_model.onnx', 'INT8 量化模型'),
    ('pcb_v4_quant_info.json', '量化报告（逐层 cosine）'),
    ('pcb_v4_original_float_model.onnx', '原始 FP32 模型'),
])

runs = pick(M / 'runs' / 'pcb_v4', [
    ('results.png', '⭐ 训练曲线'),
    ('results.csv', '逐 epoch 原始指标'),
    ('confusion_matrix_normalized.png', '⭐ 归一化混淆矩阵'),
    ('confusion_matrix.png', '混淆矩阵（原始计数）'),
    ('BoxPR_curve.png', 'PR 曲线'),
    ('BoxF1_curve.png', 'F1 曲线'),
    ('val_batch0_pred.jpg', '验证集预测可视化'),
    ('val_batch0_labels.jpg', '验证集标注真值'),
    ('train_batch0.jpg', '数据增强样例'),
])

HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>PCB 缺陷检测 · 文件导航</title>
<style>
* { box-sizing: border-box; }
body { font-family: "Microsoft YaHei", -apple-system, sans-serif; background:#0f1117; color:#d8dee9;
       margin:0; padding:28px; line-height:1.6; }
h1 { color:#7dd3fc; font-size:26px; margin:0 0 6px; }
.sub { color:#6b7280; font-size:13px; margin-bottom:26px; }
h2 { color:#a5b4fc; font-size:17px; margin:34px 0 12px; padding-left:10px; border-left:3px solid #6366f1; }
table { width:100%; border-collapse:collapse; background:#171923; border-radius:8px; overflow:hidden; font-size:13.5px; }
th { background:#1f2333; color:#9ca3af; text-align:left; padding:9px 12px; font-weight:600; font-size:12.5px; }
td { padding:8px 12px; border-top:1px solid #232838; vertical-align:middle; }
tr:hover td { background:#1c2030; }
a { color:#60d9fa; text-decoration:none; }
a:hover { text-decoration:underline; }
.sz { color:#6b7280; font-family:Consolas,monospace; text-align:right; white-space:nowrap; }
.tc { width:70px; padding:4px 8px; }
.tb { display:flex; align-items:center; justify-content:center; width:54px; height:40px;
      background:#0b0d13; border:1px solid #2d3446; border-radius:5px; overflow:hidden; }
.thumb { max-width:100%; max-height:100%; object-fit:contain; display:block; }
.card { background:#171923; border-radius:10px; padding:16px 20px; margin-bottom:22px; border:1px solid #232838; }
.tight { padding:0; overflow:hidden; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:14px; }
.kpi { background:#171923; border:1px solid #232838; border-radius:10px; padding:14px 16px; }
.kpi .v { font-size:26px; font-weight:700; color:#4ade80; font-family:Consolas,monospace; }
.kpi .k { font-size:12px; color:#6b7280; margin-top:2px; }
.warn { background:#2a1d12; border:1px solid #92400e; border-radius:8px; padding:14px 18px; font-size:13px; color:#fbbf24; }
.cmd { background:#0b0d13; border:1px solid #232838; border-radius:6px; padding:10px 14px;
       font-family:Consolas,monospace; font-size:12.5px; color:#a5b4fc; overflow-x:auto; }
.foot { color:#4b5563; font-size:12px; margin-top:40px; text-align:center; }
.badge { display:inline-block; background:#1e3a5f; color:#7dd3fc; border-radius:4px; padding:1px 7px; font-size:11.5px; }
</style>
</head>
<body>

<h1>PCB 缺陷检测 · 文件导航</h1>
<div class="sub">YOLOv8n → ONNX → INT8 → RDK X5 &nbsp;|&nbsp; 绵阳师范学院 · 电子信息科学与技术 · 陶郡奕 &nbsp;|&nbsp; 2026-11-15</div>

<div class="grid">
  <div class="kpi"><div class="v">0.993</div><div class="k">mAP50</div></div>
  <div class="kpi"><div class="v">0.730</div><div class="k">mAP50-95</div></div>
  <div class="kpi"><div class="v">3.01M</div><div class="k">参数量</div></div>
  <div class="kpi"><div class="v">18.0</div><div class="k">FPS（PC 端）</div></div>
  <div class="kpi"><div class="v">-73%</div><div class="k">INT8 体积压缩</div></div>
</div>

<h2>📊 数据对比文档</h2>
<div class="card tight"><table>
<tr><th>文件</th><th>说明</th><th>大小</th></tr>
__DOCS__
</table></div>

<h2>🖼️ 演示图（面试重点）</h2>
<div class="card tight"><table>
<tr><th class="tc"></th><th>文件</th><th>说明</th><th>大小</th></tr>
__DEMOS__
</table></div>

<h2>🧠 模型文件</h2>
<div class="card tight"><table>
<tr><th>文件</th><th>说明</th><th>大小</th></tr>
__MODELS__
</table></div>

<h2>⚙️ 量化 / 上板文件</h2>
<div class="card tight"><table>
<tr><th>文件</th><th>说明</th><th>大小</th></tr>
__BINS__
</table></div>

<h2>📈 训练过程图</h2>
<div class="card tight"><table>
<tr><th class="tc"></th><th>文件</th><th>说明</th><th>大小</th></tr>
__RUNS__
</table></div>

<h2>📁 数据集</h2>
<div class="card"><table>
<tr><th>目录</th><th>内容</th><th>数量</th><th>说明</th></tr>
<tr><td><a href="__TRAIN_IMG__">images/train</a></td><td>训练图片</td><td class="sz">10,342</td><td>模型输入原图</td></tr>
<tr><td><a href="__TRAIN_LAB__">labels/train</a></td><td>训练标注</td><td class="sz">10,342</td><td>YOLO txt 格式</td></tr>
<tr><td><a href="__VAL_IMG__">images/val</a></td><td>验证图片</td><td class="sz">1,826</td><td>全部指标计算集</td></tr>
<tr><td><a href="__VAL_LAB__">labels/val</a></td><td>验证标注</td><td class="sz">1,826</td><td>4,737 个实例</td></tr>
<tr><td><a href="__DATA_YAML__">data.yaml</a></td><td>数据集配置</td><td class="sz">—</td><td>路径 + 6 类名</td></tr>
</table></div>

<h2>🗂️ 检测结果照片</h2>
<div class="card"><table>
<tr><th>位置</th><th>数量</th><th>说明</th></tr>
<tr><td><a href="__DEMO_DIR__">results/demo/</a></td><td class="sz">3</td><td>⭐ 面试对比图（今天生成）</td></tr>
<tr><td><a href="__RESULTS_DIR__">DeepPCB-master/results/</a></td><td class="sz">253</td><td>批量检测输出 + 报告</td></tr>
<tr><td><a href="__TEST_DIR__">data/test_images/</a></td><td class="sz">233</td><td>测试原图（分布外样本）</td></tr>
<tr><td><a href="__WEB_UP__">pcb-web-app/uploads/</a></td><td class="sz">39</td><td>Web 上传原图</td></tr>
<tr><td><a href="__WEB_RE__">pcb-web-app/results/</a></td><td class="sz">39</td><td>Web 检测结果图</td></tr>
</table></div>

<h2>🚀 启动演示</h2>
<div class="card">
<div class="kpi" style="margin-bottom:14px">
  <div class="v" style="font-size:18px;color:#60d9fa">http://127.0.0.1:5000</div>
  <div class="k">服务运行中；若已关闭，双击下面的 bat 重新启动</div>
</div>
<div class="cmd">D:\\毕业设计-代码与资料\\pcb-web-app\\演示启动.bat</div>
<div style="margin-top:12px;font-size:13px;color:#9ca3af">
演示顺序：上传图 → 出检测框 → 讲明细表 → 讲量化报告 → 打开 GitHub
</div>
</div>

<h2>⚠️ 数据口径（面试别说错）</h2>
<div class="warn">
<b>1. 规则降误检率</b>　旧简历写 "90%"，实测 <b>34%</b>（FP 102→67）<br>
<b>2. 参数量</b>　是 <b>3.01M</b>，不是 6M（6MB 是文件大小）<br>
<b>3. 板端推理</b>　BIN 已编译通过，<b>板端尚未实测</b>，别说"已在板上跑实时"<br>
<b>4. mAP50=0.993 是合成数据</b>　真实相机直出会退化 —— 主动说出来是加分项
</div>

<div class="foot">PCB 项目导航 · 2026-11-15 生成　|　所有链接指向本地文件，点击用默认程序打开</div>

</body>
</html>
"""

HTML = (HTML
        .replace('__DOCS__', rows_table(docs))
        .replace('__DEMOS__', rows_table(demos, thumb=True, extra_thumb_col=True))
        .replace('__MODELS__', rows_table(models))
        .replace('__BINS__', rows_table(bins))
        .replace('__RUNS__', rows_table(runs, thumb=True, extra_thumb_col=True))
        .replace('__TRAIN_IMG__', furl(DS / 'images' / 'train'))
        .replace('__TRAIN_LAB__', furl(DS / 'labels' / 'train'))
        .replace('__VAL_IMG__', furl(DS / 'images' / 'val'))
        .replace('__VAL_LAB__', furl(DS / 'labels' / 'val'))
        .replace('__DATA_YAML__', furl(DS / 'data.yaml'))
        .replace('__DEMO_DIR__', furl(M / 'results' / 'demo'))
        .replace('__RESULTS_DIR__', furl(M / 'results'))
        .replace('__TEST_DIR__', furl(M / 'data' / 'test_images'))
        .replace('__WEB_UP__', furl(R / 'pcb-web-app' / 'static' / 'uploads'))
        .replace('__WEB_RE__', furl(R / 'pcb-web-app' / 'static' / 'results')))

out = R / 'PCB项目导航.html'
io.open(out, 'w', encoding='utf8').write(HTML)
print('written:', out)
print('  docs=%d demos=%d models=%d bins=%d runs=%d'
      % (len(docs), len(demos), len(models), len(bins), len(runs)))
