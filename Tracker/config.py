
class Policy():
    LATEST = "latest"
    BLOCK = "block"

# Models
yolo_model = "yolo11n.pt"
yolo_imgsz=640  # Reduced from 640 to improve inference speed
yolo_conf=0.25  # Increased to reduce false positives and processing time
yolo_iou=0.45   # Standard IoU threshold for better performance
yolo_classes=None  # Focus on vehicles: car, motorcycle, bus, truck
use_fp16=True
tracker_persist=True

# Throughput
target_input_fps=10
fps_ema_alpha=0.9  # EMA smoothing factor for FPS calculation (0.1-0.9)
target_proc_fps=5  # Hard cap inside Detector for processing FPS 
output_video_fps=10   # FPS for output video file (lower = slower video)

# Buffers
frames_buffer_size=10
result_buffer_size=10
writer_frame_buffer_size=10


drop_policy_frames = Policy.LATEST
drop_policy_results= Policy.LATEST
drop_policy_write_frames= Policy.LATEST


# Capture Robustness
flush_grabs_max=10
flush_max_ms=1000
max_no_frame_sec=5   # Max seconds without frames before reconnect
reconnect_delay_sec=2  # Delay between reconnection attempts

# Video
video_src: str = "video.mp4" #"http://qthttp.apple.com.edgesuite.net/1010qwoeiuryfg/sl.m3u8"
save_video=True
output_path="output.mp4"

# Geometry
fov_deg=60

# Depth estimation parameters
depth_reference_height=100  # Reference bbox height for depth calculation
depth_max_height=300       # Maximum expected bbox height (closest objects)
depth_min_height=20        # Minimum expected bbox height (farthest objects)

# GPU Configuration
device_id=0  # GPU device ID to use (0 for first GPU, 1 for second, etc.)
            # Set to -1 to use CPU only
            # The system will automatically fall back to CPU if specified GPU is not available

# Logging
verbose=True
