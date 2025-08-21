from ultralytics import YOLO
import logging
import cv2
import time
from queue import Queue, Full, Empty
import sys
import os
import threading
from typing import Any, Optional
import numpy as np

# Add parent directory to path to import tracker module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tracker import config

# Logging System (verbose)
# YOLO Configuration (yolo_model, yolo_imgsz, yolo_conf, yolo_iou, yolo_classes, use_fp16, tracker_persist)
# Streaming Configuration (target_input_fps, fps_ema_alpha, target_proc_fps)
# Video / Writer Configuration (save_video, output_path)
# Omniverse Configuration ( GPU, Geometry)


logger = logging.getLogger(__name__)

STOP_ITEM = object()

class BoundedQueue:
    """Thread-safe bounded queue with configurable drop policies"""
    
    def __init__(self, maxsize: int, drop_policy: str = "latest"):
        self.queue = Queue(maxsize=maxsize)
        self.drop_policy = drop_policy
        self._approximate_size = 0  # Track size manually for platforms without qsize()
        self._dropped_items = 0  # Track dropped items for statistics
        self._total_items = 0    # Track total items attempted
        
    def put(self, item: Any, timeout: Optional[float] = None) -> bool:
        """
        Put item in queue according to drop policy.
        Returns True if item was added, False if dropped.
        
        Drop policies:
        - "latest": Drop oldest item when full, keep newest
        - "block": Block until space available (respects timeout)
        """
        self._total_items += 1
        
        try:
            if self.drop_policy == config.Policy.BLOCK:
                self.queue.put(item, timeout=timeout)
                self._approximate_size += 1
                return True
                
            else:  # config.Policy.LATEST (default)
                try:
                    self.queue.put_nowait(item)
                    self._approximate_size += 1
                    return True
                except Full:
                    # Drop oldest item and add new one
                    try:
                        dropped_item = self.queue.get_nowait()
                        self._approximate_size = max(0, self._approximate_size - 1)
                        self._dropped_items += 1
                        if config.verbose:
                            logger.debug(f"Dropped oldest item from queue (policy: latest)")
                    except Empty:
                        pass
                    try:
                        self.queue.put_nowait(item)
                        self._approximate_size += 1
                        return True
                    except Full:
                        self._dropped_items += 1
                        if config.verbose:
                            logger.debug(f"Failed to add item after drop (policy: latest)")
                        return False
                    
        except Exception as e:
            logger.error(f"Error putting item in queue: {e}")
            self._dropped_items += 1
            return False
    
    def get(self, timeout: Optional[float] = None) -> Any:
        """Get item from queue"""
        item = self.queue.get(timeout=timeout)
        self._approximate_size = max(0, self._approximate_size - 1)
        return item
    
    def get_nowait(self) -> Any:
        """Get item from queue without blocking"""
        item = self.queue.get_nowait()
        self._approximate_size = max(0, self._approximate_size - 1)
        return item
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        return self.queue.empty()
    
    def qsize(self) -> int:
        """Get approximate queue size"""
        try:
            return self.queue.qsize()
        except NotImplementedError:
            # qsize() is not implemented on all platforms (e.g., macOS)
            # Return our manual tracking as fallback
            return self._approximate_size
        except Exception:
            return self._approximate_size
    
    def get_stats(self) -> dict:
        """Get queue statistics"""
        return {
            'size': self.qsize(),
            'dropped_items': self._dropped_items,
            'total_items': self._total_items,
            'drop_rate': self._dropped_items / max(1, self._total_items),
            'drop_policy': self.drop_policy
        }
    
    def reset_stats(self):
        """Reset queue statistics"""
        self._dropped_items = 0
        self._total_items = 0


class CameraCapture:
    
    def __init__(self, source):
        self.source = source
        self.cap = None
        self.last_frame_time = time.time()
        self.reconnect_attempts = 0
        self.source_fps = None
        
    def connect(self):
        """Connect to camera source"""
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                return False
            
            # Get source FPS
            self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.source_fps <= 0:
                self.source_fps = 25.0  # Default fallback FPS
            logger.info(f"Source FPS detected: {self.source_fps}")
            
            # Set buffer size to minimize latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return True
        except Exception as e:
            logging.error(f"Failed to connect to camera: {e}")
            return False
    
    def grab_frame(self):
        """Grab frame with flush logic for low latency"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        # Flush old frames for low latency
        flush_count = 0
        start_time = time.time()
        
        while flush_count < config.flush_grabs_max:
            if (time.time() - start_time) * 1000 > config.flush_max_ms:
                break
            
            ret = self.cap.grab()
            if not ret:
                break
            flush_count += 1
        
        # Retrieve the latest frame
        ret, frame = self.cap.retrieve()
        if ret:
            self.last_frame_time = time.time()
            return frame
        
        return None
    
    def reconnect(self):
        """Attempt to reconnect to camera source"""
        try:
            if self.cap:
                self.cap.release()
            
            logger.info(f"Attempting to reconnect to {self.source}")
            self.reconnect_attempts += 1
            
            if self.connect():
                logger.info(f"Successfully reconnected after {self.reconnect_attempts} attempts")
                self.reconnect_attempts = 0
                return True
            else:
                logger.error(f"Reconnection attempt {self.reconnect_attempts} failed")
                return False
                
        except Exception as e:
            logger.error(f"Error during reconnection: {e}")
            return False
    

class PositionEstimator:
    def __init__(self):
        

        self.running = False
        self.thread = None
        self.latest_detections = []  # Simple list to store latest detections
        self.frames_queue = BoundedQueue(maxsize=config.frames_buffer_size, drop_policy=config.drop_policy_frames)
        self.results_queue = BoundedQueue(maxsize=config.result_buffer_size, drop_policy=config.drop_policy_results)
        self.writer_queue = BoundedQueue(maxsize=config.writer_frame_buffer_size, drop_policy=config.drop_policy_write_frames)

        self.orchestrator = Orchestrator(self.frames_queue, self.results_queue, self.writer_queue, self)
        self.detector = Detector(self.frames_queue, self.results_queue)
        self.writer = Writer(self.results_queue, self.writer_queue, self.orchestrator.camera)

        self.logger = logging.getLogger(f"{__name__}.PositionEstimator")
        
    def start(self):
        logger.debug("Starting Position Estimator")
        self.running = True
        self.orchestrator.start()
        self.detector.start()
        self.writer.start()
    
    def stop(self):
        logger.debug("Stopping Position Estimator")
        self.running = False
        
        # Log queue statistics before stopping
        if config.verbose:
            self.log_queue_stats()
        
        self.orchestrator.stop()
        # Unblock consumers waiting on queues
        self.frames_queue.put(STOP_ITEM)
        self.writer_queue.put((STOP_ITEM, None))
        self.detector.stop()
        self.writer.stop()
    
    def get_detections(self):
        """Simple method to get latest detections"""
        return self.latest_detections
    
    def get_performance_stats(self):
        """Get performance and latency statistics"""
        latency_stats = self.orchestrator.get_latency_stats()
        
        return {
            "fps": {
                "input_ema": self.orchestrator.input_fps_ema,
                "target_input": config.target_input_fps,
                "target_proc": config.target_proc_fps
            },
            "latency": latency_stats,
            "stride": self.orchestrator.current_stride,
            "frames_processed": self.orchestrator.frame_count,
            "detections_processed": self.detector.process_count,
            "performance_issues": {
                "low_fps": self.orchestrator.input_fps_ema < config.target_input_fps * 0.7,
                "high_latency": latency_stats['avg_ms'] > 500 if latency_stats['samples'] > 0 else False,
                "recommendations": self._get_performance_recommendations(latency_stats)
            }
        }
    
    def _get_performance_recommendations(self, latency_stats):
        """Get performance improvement recommendations"""
        recommendations = []
        
        if latency_stats['samples'] > 0 and latency_stats['avg_ms'] > 500:
            recommendations.append("High latency detected - consider reducing target_proc_fps or increasing stride")
        
        if self.orchestrator.input_fps_ema < config.target_input_fps * 0.7:
            recommendations.append("Low input FPS - check stream source or reduce target_input_fps")
            
        if latency_stats['samples'] > 0 and latency_stats['target_500ms'] < 0.8:
            recommendations.append(f"Only {latency_stats['target_500ms']:.1%} frames meet 500ms target")
            
        return recommendations
    
    def log_queue_stats(self):
        """Log statistics for all queues"""
        frames_stats = self.frames_queue.get_stats()
        results_stats = self.results_queue.get_stats()
        writer_stats = self.writer_queue.get_stats()
        
        logger.info("=== Queue Statistics ===")
        logger.info(f"Frames Queue - Size: {frames_stats['size']}, "
                   f"Dropped: {frames_stats['dropped_items']}/{frames_stats['total_items']} "
                   f"({frames_stats['drop_rate']:.2%}), Policy: {frames_stats['drop_policy']}")
        logger.info(f"Results Queue - Size: {results_stats['size']}, "
                   f"Dropped: {results_stats['dropped_items']}/{results_stats['total_items']} "
                   f"({results_stats['drop_rate']:.2%}), Policy: {results_stats['drop_policy']}")
        logger.info(f"Writer Queue - Size: {writer_stats['size']}, "
                   f"Dropped: {writer_stats['dropped_items']}/{writer_stats['total_items']} "
                   f"({writer_stats['drop_rate']:.2%}), Policy: {writer_stats['drop_policy']}")
        logger.info("========================")
    

# Orchestrator: owns camera capture (low-latency grab→retrieve flush), dynamic
#               downsample to a target input FPS, dispatches frames to Detector via bounded
#               queue, collects latest results for Master, and (optionally) mirrors frames to Writer.
class Orchestrator:

    def __init__(self, frames_queue, results_queue, writer_queue, position_estimator):
        # Initialize camera capture
        self.camera = CameraCapture(source=config.video_src)
        if not self.camera.connect():
            logging.error("Failed to initialize camera")
            raise RuntimeError("Camera initialization failed")
        
        # Frame processing state
        self.last_input_time = 0
        self.input_fps_ema = 0.0
        self.measured_fps = 0.0
        self.frame_count = 0
        self.target_input_interval = 1.0 / config.target_input_fps
        
        # Dynamic stride calculation
        self.current_stride = 1
        self.frames_since_stride_update = 0
        self.stride_update_interval = 30  # Update stride every N frames
        
        # Latency tracking
        self.frame_timestamps = {}  # frame_id -> arrival_timestamp
        self.latency_samples = []
        self.max_latency_samples = 100
        
        # Buffers and queues
        self.frames_queue = frames_queue
        self.results_queue = results_queue
        self.writer_queue = writer_queue
        self.position_estimator = position_estimator


    def start(self):
        logger.info("Starting Orchestrator")
        self.running = True
        self.capture_thread = threading.Thread(target=self.run)
        self.capture_thread.start()


    def stop(self):
        logger.info("Stopping Orchestrator")
        self.running = False
        if self.capture_thread:
            self.capture_thread.join()
        self.camera.cap.release()

    def calculate_dynamic_stride(self):
        """Calculate optimal stride based on measured vs target FPS"""
        if self.input_fps_ema <= 0:
            return
        
        old_stride = self.current_stride
        
        # If latency is too high, reduce processing load by increasing stride
        latency_stats = self.get_latency_stats()
        if latency_stats['samples'] > 5:
            if latency_stats['avg_ms'] > 800:  # Very high latency
                self.current_stride = min(self.current_stride + 2, 5)
            elif latency_stats['avg_ms'] > 500:  # High latency
                self.current_stride = min(self.current_stride + 1, 5)
        # If latency is good and FPS is low, try to process more frames
        elif latency_stats['samples'] > 5 and latency_stats['avg_ms'] < 300 and self.input_fps_ema < config.target_input_fps * 0.9:
            self.current_stride = max(self.current_stride - 1, 1)
        # If we're getting frames much faster than target, increase stride
        elif self.input_fps_ema > config.target_input_fps * 1.3:
            self.current_stride = min(self.current_stride + 1, 5)
        
        if old_stride != self.current_stride and config.verbose:
            logger.info(f"Dynamic stride updated: {old_stride} → {self.current_stride} "
                       f"(FPS: {self.input_fps_ema:.2f}, Latency: {latency_stats['avg_ms']:.1f}ms)")

    def track_frame_latency(self, frame_id, arrival_time):
        """Track when frame arrives for latency calculation"""
        self.frame_timestamps[frame_id] = arrival_time
        
        # Clean old timestamps to prevent memory leak
        if len(self.frame_timestamps) > 200:
            oldest_keys = sorted(self.frame_timestamps.keys())[:100]
            for key in oldest_keys:
                del self.frame_timestamps[key]

    def calculate_latency(self, frame_id):
        """Calculate latency from frame arrival to now"""
        if frame_id in self.frame_timestamps:
            arrival_time = self.frame_timestamps[frame_id]
            latency_ms = (time.time() - arrival_time) * 1000
            
            # Store latency sample
            self.latency_samples.append(latency_ms)
            if len(self.latency_samples) > self.max_latency_samples:
                self.latency_samples.pop(0)
            
            return latency_ms
        return None

    def get_latency_stats(self):
        """Get latency statistics"""
        if not self.latency_samples:
            return {"avg_ms": 0, "max_ms": 0, "min_ms": 0, "samples": 0}
        
        return {
            "avg_ms": sum(self.latency_samples) / len(self.latency_samples),
            "max_ms": max(self.latency_samples),
            "min_ms": min(self.latency_samples),
            "samples": len(self.latency_samples),
            "target_500ms": sum(1 for x in self.latency_samples if x <= 500) / len(self.latency_samples)
        }

    def run(self):
        start_time = time.time()
        skip_count = 0
        
        while self.running:
            frame_start = time.time()
            frame = self.camera.grab_frame()
            
            if frame is None:
                # Check for reconnection if no frames for too long
                if time.time() - self.camera.last_frame_time > config.max_no_frame_sec:
                    logging.warning(f"No frames for {config.max_no_frame_sec}s, attempting reconnect...")
                    if not self.camera.reconnect():
                        time.sleep(config.reconnect_delay_sec)
                continue
            
            # Update FPS measurements with EMA smoothing
            self.frame_count += 1
            current_time = time.time()
            
            if self.last_input_time > 0:
                frame_interval = current_time - self.last_input_time
                instant_fps = 1.0 / frame_interval if frame_interval > 0 else 0
                
                # EMA smoothing for FPS
                if self.input_fps_ema == 0:
                    self.input_fps_ema = instant_fps
                else:
                    self.input_fps_ema = (config.fps_ema_alpha * self.input_fps_ema + 
                                         (1 - config.fps_ema_alpha) * instant_fps)
            
            self.last_input_time = current_time
            
            # Dynamic stride calculation
            self.frames_since_stride_update += 1
            if self.frames_since_stride_update >= self.stride_update_interval:
                self.calculate_dynamic_stride()
                self.frames_since_stride_update = 0
            
            # Apply dynamic stride - skip frames if needed
            if skip_count < self.current_stride - 1:
                skip_count += 1
                continue
            skip_count = 0
            
            # Track frame arrival for latency calculation
            frame_id = self.frame_count
            arrival_time = current_time
            self.track_frame_latency(frame_id, arrival_time)
            
            # Log FPS and latency periodically
            if self.frame_count % 50 == 0 and config.verbose:
                elapsed = current_time - start_time
                avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
                latency_stats = self.get_latency_stats()
                logger.info(f"Orchestrator - EMA FPS: {self.input_fps_ema:.2f}, "
                           f"Avg FPS: {avg_fps:.2f}, Frames: {self.frame_count}, "
                           f"Stride: {self.current_stride}, "
                           f"Latency: {latency_stats['avg_ms']:.1f}ms avg, "
                           f"{latency_stats['target_500ms']:.1%} under 500ms")
            
            # Add frame with metadata for latency tracking
            frame_with_metadata = (frame, frame_id, arrival_time)
            self.frames_queue.put(frame_with_metadata)
            
            # Dynamic downsample to target_input_fps as required
            time.sleep(1/config.target_input_fps)
            
            results_frame = self.results_queue.get()
            if results_frame is not None:
                # Simple detection processing
                detections = self.make_simple_detections(results_frame)
                self.position_estimator.latest_detections = detections
                
                # Calculate end-to-end latency (after detections are available)
                if isinstance(results_frame, tuple) and len(results_frame) >= 3:
                    _, result_frame_id, _ = results_frame
                    latency_ms = self.calculate_latency(result_frame_id)
                    if latency_ms and latency_ms > 500 and config.verbose:
                        logger.warning(f"High latency detected: {latency_ms:.1f}ms (target: ≤500ms)")
                self.writer_queue.put((frame, results_frame))
    
    def make_simple_detections(self, results_data):
        """Convert YOLO results to simple detection format"""
        detections = []
        
        # Handle results with metadata
        if isinstance(results_data, tuple) and len(results_data) >= 3:
            results, frame_id, arrival_time = results_data
        else:
            results = results_data
            frame_id = 0
            arrival_time = time.time()
        
        if not results:
            return detections
            
        # Handle both list and single Result object
        if isinstance(results, list):
            if len(results) == 0:
                return detections
            result = results[0]
        else:
            result = results
            
        if not result or not result.boxes:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else []
        track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else []
        
        # COCO class names (all 80 classes)
        class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
            25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
            39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
            45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
            50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
            55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
            60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
            65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
            74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
            78: 'hair drier', 79: 'toothbrush'
        }
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cls = int(classes[i]) if i < len(classes) else 2
            track_id = int(track_ids[i]) if i < len(track_ids) else i
            
            # Calculate angle_deg based on horizontal position in frame
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Calculate angle from camera center (FOV-based)
            # Assume frame dimensions (you may want to get these from actual frame)
            frame_width = 1920  # Default, should match your video resolution
            frame_height = 1080
            
            # Normalize position to [-1, 1] range
            norm_x = (center_x / frame_width) * 2 - 1
            
            # Calculate horizontal angle using FOV
            import math
            half_fov_rad = math.radians(config.fov_deg / 2)
            angle_rad = norm_x * half_fov_rad
            angle_deg = math.degrees(angle_rad)
            
            # Calculate relative depth based on bounding box size
            # Larger objects (closer) = lower depth value (0.0 = very close)
            # Smaller objects (farther) = higher depth value (1.0 = very far)
            
            # Use bbox height as primary depth indicator
            # Normalize bbox height and invert (larger = closer = lower depth)
            normalized_height = max(0, min(1, (bbox_height - config.depth_min_height) / (config.depth_max_height - config.depth_min_height)))
            relative_depth = 1.0 - normalized_height  # Invert so larger objects have lower depth
            relative_depth = max(0.0, min(1.0, relative_depth))  # Clamp to [0, 1]
            
            detection = {
                "id": track_id,
                "class": class_names.get(cls, f'unknown_{cls}'),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "angle_deg": float(angle_deg),
                "relative_depth": float(relative_depth),
                "timestamp": time.time()
            }
            detections.append(detection)
        
        return detections


        

# Detector: runs YOLOv11n detection + built-in tracker (IDs must persist) on full
#           frames; optional MiDaS relative depth; publishes structured detections to a results
#           queue.
class Detector:
    def __init__(self, frames_queue, results_queue):
        # Initialize YOLO model with GPU if available
        import torch
        
        # Configure device based on config.device_id
        if config.device_id == -1:
            # Force CPU usage
            self.device = 'cpu'
            logger.info("Using CPU as specified by device_id=-1")
        elif torch.cuda.is_available():
            if config.device_id >= torch.cuda.device_count():
                logger.warning(f"Requested GPU {config.device_id} not available. Found {torch.cuda.device_count()} GPU(s). Using CPU.")
                self.device = 'cpu'
            else:
                self.device = f'cuda:{config.device_id}'
                gpu_name = torch.cuda.get_device_name(config.device_id)
                gpu_memory = torch.cuda.get_device_properties(config.device_id).total_memory / 1024**3
                logger.info(f"Using GPU {config.device_id}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            logger.warning("CUDA not available, using CPU. Inference will be slower.")
            self.device = 'cpu'
        
        self.model = YOLO(config.yolo_model)
        self.model.to(self.device)
        self.frames_queue = frames_queue
        self.results_queue = results_queue
        self.running = False
        self.thread = None
        self.process_count = 0
        self.start_time = time.time()
        self.last_process_time = 0
        

    def start(self):
        logger.info("Starting Detector")
        self.running = True
        self.thread = threading.Thread(target=self.process_frames)
        self.thread.start()

    def stop(self):
        logger.info("Stopping Detector")
        self.running = False
        if self.thread:
            self.thread.join()

    def process_frames(self):
        while self.running:
            frame_data = self.frames_queue.get()
            if frame_data is STOP_ITEM:
                break
            if frame_data is not None:
                process_start = time.time()
                
                # Handle frame with metadata
                if isinstance(frame_data, tuple) and len(frame_data) >= 3:
                    frame, frame_id, arrival_time = frame_data
                else:
                    frame = frame_data
                    frame_id = self.process_count
                    arrival_time = process_start
                
                results = self.model.track(frame,
                                           conf=config.yolo_conf,
                                           iou=config.yolo_iou,
                                           classes=config.yolo_classes,
                                           imgsz=config.yolo_imgsz,
                                           half=config.use_fp16 and self.device != 'cpu',
                                           persist=config.tracker_persist,
                                           verbose=config.verbose,
                                           device=self.device)
                
                # Extract first result from list (track returns a list even for single frame)
                if results and isinstance(results, list):
                    result = results[0]
                else:
                    result = results
                
                # Pass results with metadata for latency tracking
                results_with_metadata = (result, frame_id, arrival_time)
                self.results_queue.put(results_with_metadata)
                self.process_count += 1
                
                # Apply target_proc_fps hard cap
                process_time = time.time() - process_start
                min_interval = 1.0 / config.target_proc_fps
                
                if process_time < min_interval:
                    sleep_time = min_interval - process_time
                    time.sleep(sleep_time)
                
                # Log processing stats periodically
                if self.process_count % 20 == 0 and config.verbose:
                    elapsed = time.time() - self.start_time
                    avg_proc_fps = self.process_count / elapsed if elapsed > 0 else 0
                    logger.info(f"Detector - Processed: {self.process_count}, "
                               f"Avg FPS: {avg_proc_fps:.2f}, "
                               f"Process time: {process_time:.3f}s")

# Writer: annotates frames with boxes/IDs/classes and writes video asynchronously;
#         must not block Detector.
class Writer:
    def __init__(self, results_queue, writer_queue, camera):
        self.results_queue = results_queue
        self.writer_queue = writer_queue
        self.camera = camera
        self.running = False
        self.thread = None
        self.video_writer = None

    def start(self):
        logger.info("Starting Writer")
        self.running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def stop(self):
        logger.info("Stopping Writer")
        self.running = False
        if self.thread:
            self.thread.join()

    def _annotate_frame(self, frame: np.ndarray, result: Any) -> np.ndarray:
        """Annotate frame with detection boxes and labels"""
        try:
            annotated = frame.copy()
            
            # Get the boxes and track IDs
            # if result.boxes and result.boxes.is_track:
            #     boxes = result.boxes.xywh.cpu()
            #     track_ids = result.boxes.id.int().cpu().tolist()

            #     # Visualize the result on the frame
            #     annotated = result.plot()
            #     track_history = {}

            #     # Plot the tracks
            #     for box, track_id in zip(boxes, track_ids):
            #         x, y, w, h = box
            #         track = track_history[track_id]
            #         track.append((float(x), float(y)))  # x, y center point
            #         if len(track) > 30:  # retain 30 tracks for 30 frames
            #             track.pop(0)

            #         # Draw the tracking lines
            #         points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            #         cv2.polylines(annotated, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                    
            if result.boxes:
                boxes = result.boxes.xyxy.cpu()
                classes = result.boxes.cls.cpu() if result.boxes.cls is not None else []
                confidences = result.boxes.conf.cpu() if result.boxes.conf is not None else []
                
                # Simple color scheme - cycle through basic colors
                colors = [
                    (0, 255, 0),      # Green
                    (255, 0, 0),      # Blue
                    (0, 0, 255),      # Red
                    (0, 255, 255),    # Yellow
                    (255, 0, 255),    # Magenta
                    (255, 255, 0),    # Cyan
                    (128, 0, 128),    # Purple
                    (255, 165, 0),    # Orange
                ]
                
                # COCO class names (same as in detection processing)
                class_names = {
                    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
                    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
                    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
                    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
                    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
                    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
                    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
                    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
                    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
                    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
                    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
                    78: 'hair drier', 79: 'toothbrush'
                }
                
                for i, box in enumerate(boxes):
                    # Get bbox coordinates
                    x1, y1, x2, y2 = [int(x) for x in box]
                    
                    # Ensure coordinates are within frame bounds
                    h, w = annotated.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # Get class and confidence
                    cls = int(classes[i]) if i < len(classes) else 2
                    conf = float(confidences[i]) if i < len(confidences) else 0.0
                    
                    # Get color by cycling through color list
                    color = colors[cls % len(colors)]
                    class_name = class_names.get(cls, f'Class{cls}')
                    
                    # Draw bounding box with thicker line
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw class label with confidence
                    label = f'{class_name}: {conf:.2f}'
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Draw label background
                    cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(annotated, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
            
            return annotated
            
        except Exception as e:
            logger.error(f"Error annotating frame: {e}")
            return frame

    def run(self):
        try:
            while self.running:
                frame, result = self.writer_queue.get()
                if frame is STOP_ITEM:
                    break
                if frame is not None:

                    if result is not None:
                        latest_detections = result[0]
                        frame = self._annotate_frame(frame, latest_detections)

                    if self.video_writer is None:
                        height, width = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        # Use target_input_fps to match processing rate
                        self.video_writer = cv2.VideoWriter(
                            config.output_path,
                            fourcc,
                            config.target_input_fps,
                            (width, height)
                        )
                        logger.info(f"Video writer initialized with {config.target_input_fps} FPS")
                    self.video_writer.write(frame)
                    # Control timing to match target processing FPS
                    time.sleep(1/config.target_input_fps)
        finally:
            if self.video_writer is not None:
                self.video_writer.release()




def main():
    logger.info("Starting Position Estimator")
    position_estimator = PositionEstimator()
    position_estimator.start()
    time.sleep(10)
    position_estimator.stop()

if __name__ == "__main__":
    main()