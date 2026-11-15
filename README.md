# PCB 缺陷检测系统

> 基于 YOLOv8n 的 PCB 缺陷检测，完成 **训练 → ONNX 导出 → INT8 量化 → 地平线 RDK X5 部署** 全链路。
> 本科毕业设计 · 绵阳师范学院 电子信息科学与技术

## 核心结果

| 指标 | 数值 | 说明 |
|:--|--:|:--|
| **mAP50** | **0.993** | 合并数据集验证集 1,826 张 / 4,737 实例 |
| mAP50-95 | 0.730 | 小目标定位是短板，属轻量化取舍（见下） |
| Precision / Recall | 0.986 / 0.989 | 训练收敛后模型自评 |
| 参数量 | **3.01 M** | YOLOv8n，640×640 输入 |
| 模型体积 | 12.2 MB → **3.3 MB** | INT8 量化，压缩 **73%** |
| 推理速度 | 18.0 FPS | PC / ONNX Runtime / CPU 基线 |

**与公开论文对比（DeepPCB 系数据集）：**

| 模型 | mAP50 | mAP50-95 |
|:--|--:|--:|
| **本工作 YOLOv8n（3.01M）** | **0.993** | 0.730 |
| Enhanced YOLOv11 (Nature Sci. Rep. 2025) | 0.989 | 0.810 |
| YOLOv11 基线 (同论文) | 0.985 | 0.792 |
| YOLOv10 (同论文) | 0.982 | 0.758 |
| DeepPCB 原论文 Faster R-CNN | 0.976 | — |

> mAP50 最高，但 mAP50-95 落后于重型网络。这是 **"轻量模型 + 640 输入 + 边缘实时"** 的主动取舍，不是训练不足 —— 换 YOLOv8m 或把输入提到 1280 都能涨点，但 X5 上就保不住实时性。

## 缺陷类别

| 索引 | 类别 | 说明 | 物理特征 |
|:--:|:--|:--|:--|
| 0 | open | 开路 / 断线 | 细长缺口 |
| 1 | short | 短路 | 异常相连 |
| 2 | mousebite | 鼠咬缺角 | 半圆形凹陷 |
| 3 | spur | 毛刺 / 突铜 | 多余突刺 |
| 4 | copper | 铜箔异常 | 不规则铜面 |
| 5 | pin-hole | 针孔 | 极小圆孔 |

## 技术路线

```
数据集(合并 12,168 张)
      ↓
YOLOv8n 训练 (180 epochs, 640×640)
      ↓
ONNX 导出 [1,3,640,640] → [1,10,8400]
      ↓
地平线 hb_mapper: 校准集 PTQ → INT8 BIN
      ↓
RDK X5 BPU 推理  +  传统 CV 后处理规则精校验
```

## 关键工作

### 1. 数据集合并与标签修复

- 合并 DeepPCB（1,275）+ PCBDatasets1（9,067）→ 训练 10,342 / 验证 1,826
- 修复原始标签 1-indexed → 0-indexed 的错位（8,515 个标注），修复前 mAP 仅 0.279

### 2. 数据增强消融

关闭 Flip（`fliplr=0, flipud=0`）后续训 180 轮：mAP50 持平，**mAP50-95 +0.010**，提升集中在 pin-hole（+0.022）、mousebite（+0.012）—— 正是方向敏感的小目标类。

### 3. INT8 量化与损失归因

| 指标 | FP32 | INT8 |
|:--|--:|--:|
| F1 | 0.975 | 0.678 |
| 体积 | 12.2 MB | 3.3 MB |

**逐层诊断定位到根因：** 分类分支 `cv2.1` 的 cosine 相似度仅 **0.911**（全模型 178 层均值 0.985，回归分支 `dfl` 0.986）。按缺陷密度分桶后：

| 每图缺陷数 | INT8 检出率 |
|--:|--:|
| 1–3 个 | **91–94%** |
| 4+ 个 | **40%** |

→ 结论：INT8 在 **密集小目标** 场景下分类头量化饱和，稀疏场景几乎无损。这是真实缺口，也是后续 QAT / 混合精度的切入方向。

### 4. 深度学习 + 传统 CV 融合后处理

9 条规则，针对模型最常犯的误检类型，用物理先验做精校验（DL 负责召回，CV 负责精校验）。

**验证集 1,826 张实测：**

| 指标 | 无规则 | 有规则 | 变化 |
|:--|--:|--:|--:|
| 误检 FP | 102 | **67** | **−34%** |
| Precision | 0.978 | **0.986** | +0.7pp |
| Recall | 0.971 | 0.969 | −0.2pp |

**按类别拆解（关键发现）：**

| 类别 | 滤掉误检 | 滤掉真检 | 规则精度 |
|:--|--:|--:|--:|
| **open** | **22** | 3 | **88%** |
| short | 1 | 3 | 25% |
| 其他 4 类 | 12 | 3 | 75–100% |

净收益高度集中在 **open 类** —— 过孔是圆形、开路缺陷是细长缺口，这个物理判据最可靠。**short 类规则精度仅 25%，说明判据在该类上不可靠，是下一步要改的地方。**

## 快速开始

```bash
pip install -r requirements.txt

# 批量检测（默认跑 data/test_images）
python scripts/detect.py

# 单张图 / 摄像头
python scripts/detect.py --source path/to/pcb.jpg
python scripts/detect.py --source 0

# 关闭后处理规则，对比原始模型输出
python scripts/detect.py --no-rules

# 性能测试
python scripts/benchmark.py --imgs 100

# Web 演示
cd ../pcb-web-app && python app.py    # http://127.0.0.1:5000
```

> **clone 后能直接跑什么：** `detect.py`、`benchmark.py` —— 仓库自带模型权重和 233 张测试图，无需额外准备。
> **`eval_*.py` 需要数据集**（要 GT 标注才能算 P/R），数据集不随仓库分发，见下节。

> **路径说明：** 所有脚本通过 `scripts/paths.py` 解析路径，**不写死盘符**。项目整体搬家后无需改任何脚本；数据集默认在同级目录 `DeepPCB_YOLO/`，可用环境变量覆盖。

## 数据集

数据集**不随本仓库分发** —— 训练集 10,342 张（约 1 GB），不适合放进 git。

需要重新训练或跑 `eval_*.py` 时，自行准备并放到与 `DeepPCB-master` **同级**的 `DeepPCB_YOLO/`：

```
毕业设计-代码与资料/
├─ DeepPCB-master/          ← 本仓库
└─ DeepPCB_YOLO/            ← 数据集（自行准备）
   ├─ images/train/         10,342 张
   ├─ images/val/           1,826 张
   ├─ labels/train/
   └─ labels/val/
```

| 来源 | 图片数 | 说明 |
|:--|--:|:--|
| [DeepPCB](https://github.com/Charmve/DeepPCB) | 1,500 | 开源 PCB 缺陷数据集，6 类 |
| PCBDatasets1 | 9,067 | 6 类 PCB 缺陷数据集 |

合并前需将 PCBDatasets1 的类别号映射为本文类别索引（缺失标注需转 0-indexed）：

| PCBDatasets1 | → | 本文索引 |
|:--|:--:|:--|
| missing_hole | → | 5 pin-hole |
| mouse_bite | → | 2 mousebite |
| open_circuit | → | 0 open |
| short | → | 1 short |
| spur | → | 3 spur |
| copper | → | 4 copper |

路径配置见 [`data/data.yaml`](data/data.yaml)（默认相对路径 `../DeepPCB_YOLO`）。数据集缺失时，`eval_*.py` 会打印明确的放置指引而不是抛异常。

## 项目结构

```
├─ scripts/
│  ├─ paths.py            统一路径解析 + 中文路径读写
│  ├─ detect.py           PC 推理（ONNX）+ 9 条后处理规则
│  ├─ benchmark.py        单帧耗时 / FPS 测试
│  ├─ export.py           ONNX 导出
│  ├─ convert_bin.py      ONNX → 地平线 BIN（Docker）
│  ├─ train_pcb_v*.py     训练脚本
│  └─ eval_*.py           精度 / 混淆矩阵 / 规则有效性评估
├─ model/
│  ├─ best.pt             pcb_v4 训练权重
│  ├─ best.onnx           ONNX 导出
│  └─ pcb_v4_noflip.onnx  消融实验版
├─ config/
│  ├─ pcb_v4_nchw.yaml    量化配置（march=bayes-e）
│  └─ pcb_v4_bin/         量化产物：pcb_v4.bin + INT8 ONNX
├─ runs/pcb_v4/           训练曲线、混淆矩阵、PR 曲线
├─ results/               实验报告 + 演示对比图
├─ data/test_images/      测试图（分布外样本）
└─ 实验日志.md            完整实验记录
```

## 实验报告

| 文档 | 内容 |
|:--|:--|
| [数据对比总表](results/数据对比总表.md) | 全部实验数据汇总（HTML 版可直接打印） |
| [量化对比报告](results/量化对比报告_FP32_vs_INT8.md) | FP32 vs INT8 精度、逐层 cosine、根因分析 |
| [实验日志](实验日志.md) | 完整过程记录、踩坑与修复 |

## 已知局限

1. **mAP50=0.993 是合成数据集的天花板** —— 模型尚未见过真实相机直出图像，部署后分布外泛化会下降。规划路线：**先部署跑实拍 → 只针对翻车样本做增量训练**，而不是盲目扩标注。
2. **板端实测待补** —— BIN 已编译通过（`config/pcb_v4_bin/pcb_v4.bin`），RDK X5 上的 FPS / 延迟数据尚未采集。
3. **密集缺陷场景量化损失大** —— 见上文 40% 检出率，后续走 QAT 或分类分支混合精度。
4. **后处理规则阈值基于 PC 测试集** —— 换真实相机后需按实拍置信度分布重新标定。

## 环境

- 训练：Windows 11 + Python 3.13 + ultralytics 8.4.83 + RTX 4050 6GB
- 量化：Docker `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310`，march=`bayes-e`
- 部署：地平线 RDK X5（BPU，10 TOPS）

## License

见 [LICENSE](LICENSE)
