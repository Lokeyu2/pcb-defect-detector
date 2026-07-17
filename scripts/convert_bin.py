"""
ONNX -> 地平线BIN转换（Docker）
用法: python scripts/convert_bin.py

前提: Docker Desktop已启动，地平线工具链镜像已拉取
"""
import os
import subprocess
import sys

# 路径
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_MNT = '/mnt/d/DeepPCB-master'

# Docker命令
CMD = f'''
docker run --rm -v {PROJECT_DIR}:{PROJECT_MNT} \\
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310 \\
  bash -c "cd {PROJECT_MNT} && hb_mapper makertbin --config config/pcb_yolov8_nchw.yaml --model-type onnx"
'''

if __name__ == '__main__':
    print("正在转换ONNX->BIN...")
    print("确保: 1) Docker Desktop已启动  2) config/pcb_yolov8_nchw.yaml已配置")
    ret = subprocess.call(CMD, shell=True)
    if ret == 0:
        print("转换成功! BIN文件在 model/bin_output/")
    else:
        print("转换失败, 检查日志")
