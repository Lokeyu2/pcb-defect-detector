"""统一路径解析。

所有脚本的路径都从这里取，不写死盘符 —— 项目整体搬家（如 D:\\DeepPCB-master
→ D:\\毕业设计-代码与资料\\DeepPCB-master）后无需改任何脚本。

另外提供支持中文路径的 imread/imwrite：cv2.imread 在 Windows 上遇到中文路径
会直接返回 None，必须走 np.fromfile + imdecode。
"""
from pathlib import Path

import cv2
import numpy as np

# scripts/paths.py → scripts → DeepPCB-master
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# DeepPCB-master 与 DeepPCB_YOLO 平级
DATASET_ROOT = PROJECT_ROOT.parent / 'DeepPCB_YOLO'

MODEL_DIR = PROJECT_ROOT / 'model'
ONNX_PATH = MODEL_DIR / 'best.onnx'
ONNX_NOFLIP = MODEL_DIR / 'pcb_v4_noflip.onnx'
BIN_DIR = PROJECT_ROOT / 'config' / 'pcb_v4_bin'
RESULTS_DIR = PROJECT_ROOT / 'results'
TEST_IMAGES = PROJECT_ROOT / 'data' / 'test_images'
DATA_YAML = PROJECT_ROOT / 'data' / 'data.yaml'

VAL_IMG = DATASET_ROOT / 'images' / 'val'
VAL_LAB = DATASET_ROOT / 'labels' / 'val'
TRAIN_IMG = DATASET_ROOT / 'images' / 'train'
TRAIN_LAB = DATASET_ROOT / 'labels' / 'train'


def imread(path):
    """读图，支持中文路径。失败返回 None。"""
    path = Path(path)
    if not path.is_file():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path, img):
    """写图，支持中文路径。返回是否成功。"""
    path = Path(path)
    ext = path.suffix if path.suffix else '.jpg'
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))
    return True


def wsl_path(path):
    """D:\\a\\b → /mnt/d/a/b，给 Docker / WSL 里的地平线工具链用。"""
    p = Path(path).resolve()
    drive = p.drive.rstrip(':').lower()
    rest = str(p).replace('\\', '/').split(':', 1)[-1]
    return f'/mnt/{drive}{rest}'
