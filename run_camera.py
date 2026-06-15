# run_camera.py
import cv2
from fatigue_detector import FatigueSession

session = FatigueSession()
cap = cv2.VideoCapture(0)

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    annotated_frame, alert = session.process_frame(frame, frame_id)
    if alert:
        print(f"[WARNING] {alert}")
    cv2.imshow('Driver Fatigue Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_id += 1

cap.release()
cv2.destroyAllWindows()