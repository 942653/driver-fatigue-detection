# Driver Fatigue Detection with YOLOv8 & FastAPI

基于 YOLOv8 的疲劳驾驶实时检测 Web 应用，支持图片、视频上传和浏览器摄像头流。

## Features
- 自定义 YOLOv8 模型，检测 `microsleep`, `distraction`, `phone_use` 等状态
- 导出 ONNX 模型，加速推理
- FastAPI 提供 RESTful API 和 WebSocket
- 图片/视频上传推理，返回 JSON 或带标注的视频
- 浏览器实时摄像头流，自动叠加警告信息
- 疲劳帧计数 + 冷却机制，防止误报

## Demo
![demo](demo.gif)

## Quick Start
### 1. 克隆仓库
...

### 2. 安装依赖
...

### 3. 下载模型
将训练好的 `best.onnx` 放入项目根目录

### 4. 启动服务
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000