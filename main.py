from fastapi import FastAPI, File, UploadFile
import cv2, numpy as np
from fatigue_detector import model  # 直接使用已加载的模型
import tempfile, os
from fastapi.responses import FileResponse
from fatigue_detector import FatigueSession
from fastapi import WebSocket, WebSocketDisconnect
import base64

app = FastAPI()

@app.post("/predict/image") #单张图片上传推理
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 简单推理，不维护疲劳计数器
    results = model(img, imgsz=960, conf=0.05, iou=0.45, verbose=False)
    
    # 提取检测结果
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "class": model.names[cls_id],
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist()
        })
    has_microsleep = any(d['class'] == 'microsleep' for d in detections)
    
    return {
        "filename": file.filename,
        "microsleep_detected": has_microsleep,
        "detections": detections
    }



@app.post("/predict/video")   #视频文件上传
async def predict_video(file: UploadFile = File(...)):
    # 保存上传视频到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(await file.read())
        input_path = tmp_in.name

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = input_path.replace(".mp4", "_out.mp4")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    session = FatigueSession()
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        annotated_frame, alert = session.process_frame(frame, frame_id)
        out.write(annotated_frame)
        frame_id += 1

    cap.release()
    out.release()
    return FileResponse(output_path, media_type="video/mp4", filename="processed.mp4")



@app.websocket("/ws")   #实时摄像头流
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = FatigueSession()
    frame_id = 0
    try:
        while True:
            # 接收前端发来的 base64 编码图像
            data = await websocket.receive_text()
            img_bytes = base64.b64decode(data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 调用疲劳检测
            annotated_frame, alert = session.process_frame(frame, frame_id)
            if alert:
                print(f"Alert sent to client: {alert}")

            # 编码后发回
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            b64_str = base64.b64encode(buffer).decode('utf-8')
            await websocket.send_text(b64_str)
            frame_id += 1
    except WebSocketDisconnect:
        print("Client disconnected")