"""
ONNX -> 地平线BIN转换（Docker）
用法: python scripts/convert_bin.py

前提: Docker Desktop已启动，地平线工具链镜像已拉取
"""
import os
import subprocess
import sys

import paths

# 路径
PROJECT_DIR = str(paths.PROJECT_ROOT)
PROJECT_MNT = paths.wsl_path(paths.PROJECT_ROOT)

# Docker命令
CMD = f'''
docker run --rm -v {PROJECT_DIR}:{PROJECT_MNT} \\
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310 \\
  bash -c "cd {PROJECT_MNT} && hb_mapper makertbin --config config/pcb_v4_nchw.yaml --model-type onnx"
'''

if __name__ == '__main__':
    print("正在转换ONNX->BIN...")
    print(f"  项目目录: {PROJECT_DIR}")
    print(f"  容器挂载: {PROJECT_MNT}")
    print("确保: 1) Docker Desktop已启动  2) config/pcb_v4_nchw.yaml已配置")
    ret = subprocess.call(CMD, shell=True)
    if ret == 0:
        print("转换成功! BIN文件在 config/pcb_v4_bin/")
    else:
        print("转换失败, 检查日志")
