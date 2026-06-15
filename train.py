# train_optimized.py
from ultralytics import YOLO

def main():
    model = YOLO('yolov8s.pt')   # 换成小模型

    model.train(
        data='data.yaml',
        epochs=100,
        imgsz=960,
         batch=8,
        degrees=20.0,            # 默认 0.0，现在允许 ±20° 旋转
        shear=10.0,              # 默认 0.0，增加错切变形，模拟非正面
        perspective=0.0005,      # 默认 0.0，轻微透视变换
        fliplr=0.5,              # 保持左右翻转
        mosaic=0.5,              # mosaic 虽然强，但可能切碎小目标，可适当降低                
        close_mosaic=0,          # 最后几轮关闭
        lr0=0.005,
        lrf=0.001,
        name='driver_risk_opt',
        patience=15,
        device='cuda',
        workers=0
    )

if __name__ == '__main__':
    main()