# Driver Fatigue Detection with YOLOv8 & FastAPI

基于 YOLOv8 的疲劳驾驶实时检测 Web 应用，支持图片、视频上传和浏览器摄像头流。

# 🚗 Driver Fatigue Detection System

基于 YOLOv8 与 FastAPI 的驾驶员疲劳实时检测系统，支持图片、视频上传推理以及浏览器摄像头实时推流。模型可识别 **microsleep（微睡眠）**、**distraction（分心）**、**phone_use（使用手机）** 等危险驾驶行为，并触发帧级疲劳报警机制。

---

## 📸 效果展示

> ⚠️ 请替换为你实际录制的 Demo GIF / 截图

![API 测试结果](dscreenshot.png)

---

## ✨ 主要特性

- **自定义 YOLOv8 模型**：在自标注数据集上微调，检测 4 类驾驶行为
- **ONNX 导出与优化**：模型导出为 ONNX 格式，实现跨平台高效推理
- **FastAPI RESTful API**：
  - `POST /predict/image` 上传图片，返回 JSON 检测结果
  - `POST /predict/video` 上传视频，返回带标注框的处理后视频
- **WebSocket 实时流**：浏览器采集摄像头帧，后端实时推理并返回标注图像
- **疲劳警报机制**：连续检测到 microsleep 达到阈值触发警报，并内置冷却时间防止重复报警
- **状态可视化**：在视频画面上叠加当前计数器、警报文字等 UI 信息
- **工程化代码结构**：核心检测逻辑模块化，支持本地摄像头脚本与 Web 服务共用

---

## 🛠️ 技术栈

`Python` `PyTorch` `Ultralytics YOLOv8` `ONNX Runtime` `FastAPI` `OpenCV` `WebSocket` `Uvicorn`

---

## 📁 项目结构
├── main.py # FastAPI 应用入口
├── fatigue_detector.py # 检测逻辑封装（模型加载、疲劳判断、画框）
├── run_camera.py # 本地摄像头实时推理脚本
├── train.py # YOLOv8 训练脚本
├── export.py # 模型导出为 ONNX（位于 runs/export.py）
├── split_dataset.py # 数据集划分工具
├── data.yaml # 训练配置文件（类别、路径）
├── requirements.txt # Python 依赖（请自行生成）
├── .gitignore
├── README.md
├── runs/ # 训练结果（实验记录、指标图表）
│ └── detect/
│ └── driver_risk_opt/ # 最优模型训练结果
└── dataset/ # 数据集（已忽略，需自行下载）

---

## 🚀 快速开始

### 1. 环境配置

建议使用 Python 3.8+ 并创建虚拟环境。

```bash
# 克隆仓库
git clone https://github.com/942653/driver-fatigue-detection.git
cd driver-fatigue-detection

# 创建并激活虚拟环境（可选）
conda create -n ai_env python=3.10
conda activate ai_env

# 安装依赖
pip install -r requirements.txt
#主要依赖（供手动安装参考）
fastapi
uvicorn[standard]
python-multipart
opencv-python-headless
ultralytics
onnxruntime   # 如需 ONNX 推理
---------------------------------------------------------------------------------------------------------------------------------
由于仓库体积限制，预训练权重需单独下载。

📎 百度网盘 / [通过网盘分享的文件：疲劳驾驶检测权重文件
链接: https://pan.baidu.com/s/19PkPNFz1FcFZrIMxvLo0-A 提取码: 3468]

下载后将 best.pt 或 best.onnx 放置在项目根目录。
---------------------------------------------------------------------------------------------------------------------------------
启动 Web 服务
在终端运行  uvicorn main:app --reload --host 0.0.0.0 --port 8000
打开浏览器访问：
Swagger 文档 & 在线测试：http://127.0.0.1:8000/docs
ReDoc 文档：http://127.0.0.1:8000/redoc
---------------------------------------------------------------------------------------------------------------------------------
API 接口
POST /predict/image - 单张图片推理
上传一张图片（JPEG/PNG），返回检测到的所有目标及是否包含微睡眠行为。
请求示例（Python）：
import requests

with open('test.jpg', 'rb') as f:
    r = requests.post('http://127.0.0.1:8000/predict/image', files={'file': f})
print(r.json())
响应示例：
{
  "filename": "test.jpg",
  "microsleep_detected": true,
  "detections": [
    {
      "class": "microsleep",
      "confidence": 0.92,
      "bbox": [120.3, 80.5, 200.1, 150.9]
    },
    ...
  ]
}

POST /predict/video - 视频文件推理
上传视频（MP4 等），返回带标注框和状态信息的处理后的视频文件。
WS /ws - 摄像头实时流
建立 WebSocket 连接后，前端发送 Base64 编码的 JPEG 帧，服务端返回处理后的 Base64 图像。

示例前端代码见下方「实时摄像头测试」。
实时摄像头测试（WebSocket）
在 templates/ 目录下创建一个 index.html，内容如下（可快速测试）：
<!DOCTYPE html>
<html>
<body>
    <video id="video" autoplay></video>
    <canvas id="canvas" style="display:none;"></canvas>
    <img id="result" />
    <script>
        const ws = new WebSocket("ws://localhost:8000/ws");
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');

        // 打开摄像头
        navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
            video.srcObject = stream;
            video.play();
        });

        // 每 100ms 发送一帧
        setInterval(() => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);
            const data = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
            ws.send(data);
        }, 100);

        ws.onmessage = (e) => {
            document.getElementById('result').src = "data:image/jpeg;base64," + e.data;
        };
    </script>
</body>
</html>

将此文件放入 templates/ 后，可通过 http://127.0.0.1:8000/static/index.html 或单独启动 HTTP 服务访问（需注意 CORS，或直接用 FastAPI 挂载静态文件）。

---------------------------------------------------------------------------------------------------------------------------------
模型训练
数据集
采用公开驾驶员行为数据集（或自建数据集），包含类别：
Mendeley Data平台上的Driver Risk Behavior Dataset

microsleep

distraction

phone_use

normal（正常驾驶）

数据配置见 data.yaml。



训练命令
bash
python train.py

或直接使用 Ultralytics CLI：
bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=960

训练结果
最优模型保存在 runs/detect/driver_risk_opt/weights/best.pt
训练指标（部分）：

指标	    值
mAP50	    0.90801
mAP50-95	0.50519
Precision	0.85944
Recall	    0.83874


导出 ONNX
bash
python runs/export.py
生成 best.onnx 文件，可直接用于 FastAPI 推理。

📈 待优化 & 贡献方向
添加前端可视化页面（Vue/React）

支持多路视频并发推理

疲劳检测历史记录存储（SQLite/MySQL）

使用 TensorRT 进一步加速推理

欢迎提 Issue 和 PR！

📄 许可证
本项目仅用于学习与演示，数据集和模型权重请遵循原始版权声明。

👨‍💻 作者
GitHub: @942653

项目链接: https://github.com/942653/driver-fatigue-detection

