# export.py
from ultralytics import YOLO

# 加载已训练好的模型
model = YOLO(r'C:\Users\34051\Desktop\Fatigue_Detection_Project\runs\detect\driver_risk_opt\weights\best.pt')

# 导出为 ONNX
model.export(format='onnx', dynamic=True)