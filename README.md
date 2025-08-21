# Real-Time Multi-Process Detection + Omniverse 3D Map

A real-time system that ingests live video streams, detects & tracks objects with YOLOv11n, runs as four processes with configurable buffers, exposes a REST API, and renders detections as live markers on a 3D city map in NVIDIA Omniverse.

## How to Run the API + Processes

### Windows

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the complete demo:
```bash
run_local_demo.bat
```

This will:
- Start the API (Master process)
- Call `/start` to initialize all processes
- Launch the Omniverse visualization
- Show `/read` output

### Connect to Detection Stream

The Omniverse app automatically connects to the detection API at `http://localhost:8000`. Use the UI controls to:
- Start/Stop detection system
- Read current detections
- Adjust coordinate mapping parameters in real-time


###########################################


### Manual Process Control

Start API server:
```bash
python3 API/app.py
```

Start detection processes:
```bash
curl -X POST http://localhost:8000/start
```

Stop detection processes:
```bash
curl -X POST http://localhost:8000/stop
```

Read latest detections:
```bash
curl http://localhost:8000/read
```

## Video Stream Configuration

Place your test video in the project root and update `config.py`:
```python
video_source = "test_video.mp4"
```

## Omniverse Scene Setup

### Prerequisites

1. Install NVIDIA Omniverse Kit
2. Ensure `kit.exe` is in your PATH or update the path in `run_omniverse_viz.bat`

### Launch Visualization

1. Start the API and detection processes (see above)
2. Run the Omniverse visualization:

**Windows:**
```bash
run_omniverse_viz.bat
```

### Connect to Detection Stream

The Omniverse app automatically connects to the detection API at `http://localhost:8000`. Use the UI controls to:
- Start/Stop detection system
- Read current detections
- Adjust coordinate mapping parameters in real-time

## Performance Notes

### Test Configuration
- **GPU Model**: NVIDIA RTX 4090
- **Input FPS**: ~30 FPS 
- **Target Input FPS**: 8 FPS 
- **Measured Processed FPS**: ~7.8 FPS


### Latency Optimization

For minimal latency (≤500ms target):
- Set `drop_policy_frames = "latest"`
- Set `frames_buffer_size = 1`
- Use `yolo_imgsz = 384` or smaller
- Enable `use_fp16 = True`

## Asset Source & License

### 3D City Asset

**Source**: Nvidia Sample USD Pack for city USD
**Location**: `Assets/city.usd`


