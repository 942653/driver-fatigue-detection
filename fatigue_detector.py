# fatigue_detector.py
import cv2
from ultralytics import YOLO

# ========== 可调参数 ==========
MODEL_PATH = r'C:\Users\34051\Desktop\Fatigue_Detection_Project\runs\detect\driver_risk_opt\weights\best.onnx'
MICROSLEEP_THRESH = 15
ALERT_COOLDOWN = 30
# ==============================

# 全局只加载一次模型
model = YOLO(MODEL_PATH)
print('模型加载成功，类别:', model.names)

class FatigueSession:
    """为每个视频流/会话维护独立状态"""
    def __init__(self):
        self.microsleep_counter = 0
        self.last_alert_frame = -ALERT_COOLDOWN

    def process_frame(self, frame, frame_id=0):
        """
        处理单帧，返回带标注的图像和警报信息
        frame_id: 当前帧序号，用于冷却计时（如果无此值可用时间戳代替）
        """
        results = model(frame, imgsz=960, conf=0.05, iou=0.45, verbose=False)
        annotated_frame = results[0].plot()

        # --- 检测当前帧的类别 ---
        has_microsleep = False
        distraction_info = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            if name == 'microsleep':
                has_microsleep = True
            elif name in ['distraction', 'phone_use']:
                distraction_info.append(name)

        # --- 更新计数器 ---
        self.microsleep_counter = self.microsleep_counter + 1 if has_microsleep else 0

        # --- 警报判断 ---
        alert_msg = ""
        if self.microsleep_counter >= MICROSLEEP_THRESH:
            alert_msg = "FATIGUE ALERT: Microsleep detected!"
            self.microsleep_counter = 0
        if alert_msg and (frame_id - self.last_alert_frame) < ALERT_COOLDOWN:
            alert_msg = ""   # 冷却中，不重复报警
        if alert_msg:
            self.last_alert_frame = frame_id

        # --- 在画面上叠加信息 ---
        cv2.putText(annotated_frame, f"Microsleep: {self.microsleep_counter}/{MICROSLEEP_THRESH}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if distraction_info:
            cv2.putText(annotated_frame, f"Distraction: {', '.join(distraction_info)}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        if alert_msg:
            cv2.putText(annotated_frame, alert_msg, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return annotated_frame, alert_msg