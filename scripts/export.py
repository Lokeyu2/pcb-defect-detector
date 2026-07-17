"""
导出ONNX和TorchScript
用法: python scripts/export.py
"""
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('model/best.pt')
    model.export(format='onnx', imgsz=640, half=False, simplify=True)
    model.export(format='torchscript', imgsz=640)
    print("导出完成: model/best.onnx, model/best.torchscript.pt")
