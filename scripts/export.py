"""
导出ONNX和TorchScript
用法: python scripts/export.py [opset]
默认opset=19(onnxruntime推理用); 地平线工具链用 opset=11
"""
import sys
from ultralytics import YOLO

if __name__ == '__main__':
    opset = int(sys.argv[1]) if len(sys.argv) > 1 else 19
    model = YOLO('model/best.pt')
    # opset=11: 地平线工具链 v1.2.8 最大只支持 opset 11
    model.export(format='onnx', imgsz=640, half=False, simplify=True, opset=opset)
    print(f"导出完成: model/best.onnx (opset={opset})")
