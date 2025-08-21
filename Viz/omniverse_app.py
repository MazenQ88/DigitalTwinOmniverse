import omni.kit.app
import omni.ui as ui
import omni.usd
import omni.usd.commands
import omni.kit.commands
import omni.kit.actions.core as actions
from omni.kit.viewport.window import ViewportWindow
from pxr import Usd, UsdGeom, Gf, Sdf
import sys
import os
import asyncio
import aiohttp
import json
import time
import math
import logging
import carb

# Import config module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tracker import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectionVisualizer:
    """Handles 3D visualization of detections in Omniverse"""
    
    def __init__(self):
        self.stage = None
        self.detection_prims = {}  # track_id -> prim_path
        self.api_url = "http://127.0.0.1:8000"
        self.last_update_time = 0
        self.update_interval = 0.1  # 100ms update interval
        
        # Class colors (matching detection system)
        self.class_colors = {
            'person': (0.0, 1.0, 0.0),      # Green
            'bicycle': (1.0, 0.0, 0.0),     # Blue (BGR to RGB)
            'car': (0.0, 0.0, 1.0),         # Red
            'motorcycle': (0.0, 1.0, 1.0),  # Yellow
            'airplane': (1.0, 0.0, 1.0),    # Magenta
            'bus': (1.0, 1.0, 0.0),         # Cyan
            'train': (0.5, 0.0, 0.5),       # Purple
            'truck': (1.0, 0.6, 0.0),       # Orange
        }
        
        # Coordinate mapping parameters
        self.image_width = 1920  # Assumed image width (should match your camera/video)
        self.image_height = 1080  # Assumed image height
        self.ground_y = 0.0  # Ground plane Y coordinate
        
        # Depth estimation parameters (tune these for better positioning)
        self.reference_bbox_height = 100  # Reference car height in pixels at reference distance
        self.reference_distance = 50     # Reference distance in world units
        self.min_distance = 10          # Minimum distance clamp
        self.max_distance = 200         # Maximum distance clamp
        self.vehicle_height_offset = 2  # Height above ground for vehicles
        
    def setup_stage(self, usd_file_path=None):
        """Initialize USD stage and load city scene"""
        try:
            # Get the USD context
            self.context = omni.usd.get_context()
            
            # Load USD file if provided
            if usd_file_path:
                # Convert to absolute path
                usd_path = os.path.abspath(usd_file_path)
                logger.info(f"Loading USD file: {usd_path}")
                
                if os.path.exists(usd_path):
                    self.context.open_stage(usd_path)
                    logger.info(f"Opened USD stage: {usd_path}")
                else:
                    logger.warning(f"USD file not found: {usd_path}, creating new stage")
                    self.context.new_stage()
            else:
                # Create new stage if no file provided
                logger.info("No USD file provided, creating new stage")
                self.context.new_stage()
            
            # Wait for stage to be ready
            self.stage = None
            self.stage_ready = False
            
            # Subscribe to stage events
            event_stream = self.context.get_stage_event_stream()
            self.stage_event_sub = event_stream.create_subscription_to_pop(
                self._on_stage_event, name="detection_viz_stage_events"
            )
            
            # Give it a moment to load
            import time
            time.sleep(0.5)
            
            # Get the current stage
            self.stage = self.context.get_stage()
            if not self.stage:
                logger.error("No USD stage available after loading")
                return False
            
            # Ensure default prim exists
            if not self.stage.GetDefaultPrim():
                world = UsdGeom.Xform.Define(self.stage, Sdf.Path("/World")).GetPrim()
                self.stage.SetDefaultPrim(world)
                logger.info("Created default prim /World")
            
            
            # Create detection root
            self.detection_root = UsdGeom.Xform.Define(self.stage, "/Detections")
            
            logger.info("USD stage setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup stage: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _on_stage_event(self, event):
        """Handle stage events"""
        from omni.usd import StageEventType
        
        if event.type == StageEventType.OPENED:
            self.stage = self.context.get_stage()
            if self.stage:
                self.stage_ready = True
                root = self.stage.GetRootLayer()
                logger.info(f"Stage opened - Layer: {root.identifier if root else '<no root>'}")
                
                # Log stage info
                dp = self.stage.GetDefaultPrim()
                logger.info(f"Default prim: {dp.GetPath() if dp else '<none>'}")
                roots = [p.GetPath().pathString for p in self.stage.GetPseudoRoot().GetChildren()]
                logger.info(f"Root children: {roots}")
    
    
    def screen_to_world_coordinates(self, bbox, angle_deg):
        """Convert screen coordinates to world coordinates using proper perspective projection"""
        try:
            x1, y1, x2, y2 = bbox
            
            # Calculate center of bounding box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Calculate bounding box size (for depth estimation)
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Normalize screen coordinates to [-1, 1] range
            norm_x = (center_x / self.image_width) * 2 - 1
            norm_y = (center_y / self.image_height) * 2 - 1
            
            # Convert FOV from degrees to radians
            import math
            fov_rad = math.radians(config.fov_deg)
            
            # Improved depth estimation using multiple factors
            # 1. Bbox height (primary indicator)
            height_based_distance = (self.reference_bbox_height / max(bbox_height, 1)) * self.reference_distance
            
            # 2. Bbox position in frame (objects lower in frame are typically closer)
            # Bottom of frame = closer, top = farther
            vertical_factor = 1.0 + (norm_y * 0.3)  # Objects at bottom appear closer
            
            # 3. Combined distance estimation
            estimated_distance = height_based_distance * vertical_factor
            estimated_distance = max(self.min_distance, min(self.max_distance, estimated_distance))
            
            # Calculate world coordinates
            # For a camera looking down the Z-axis with Y up:
            half_fov_tan = math.tan(fov_rad / 2)
            
            # X: horizontal position (left/right)
            world_x = norm_x * estimated_distance * half_fov_tan
            
            # Z: depth (forward/back from camera)
            # Use negative Z so positive Z moves away from camera (standard convention)
            world_z = -estimated_distance
            
            # Y: height above ground
            # Adjust based on vertical position in frame and estimated distance
            base_height = self.ground_y + self.vehicle_height_offset
            
            # Objects higher in the frame might be on elevated surfaces
            height_adjustment = -norm_y * 2.0  # Higher in frame = higher Y
            world_y = base_height + height_adjustment
            
            # Ensure vehicles stay reasonably close to ground
            world_y = max(self.ground_y, min(world_y, self.ground_y + 10))
            
            logger.debug(f"Converted bbox {bbox} -> world ({world_x:.1f}, {world_y:.1f}, {world_z:.1f})")
            logger.debug(f"  center: ({center_x:.1f}, {center_y:.1f}), norm: ({norm_x:.3f}, {norm_y:.3f})")
            logger.debug(f"  bbox_height: {bbox_height:.1f}, estimated_distance: {estimated_distance:.1f}")
            
            return (world_x, world_y, world_z)
            
        except Exception as e:
            logger.error(f"Error in coordinate conversion: {e}")
            return (0, 5, 0)
    
    def create_detection_marker(self, track_id, class_name, bbox, angle_deg, relative_depth=0.5):
        """Create or update a detection marker"""
        try:
            prim_path = f"/Detections/Detection_{track_id}"
            
            # Get world coordinates
            world_pos = self.screen_to_world_coordinates(bbox, angle_deg)
            
            # Get class color and modify based on depth (closer = brighter)
            base_color = self.class_colors.get(class_name, (0.5, 0.5, 0.5))
            # Adjust brightness based on depth (closer objects are brighter)
            depth_factor = 1.0 - (relative_depth * 0.5)  # Range: 0.5 to 1.0
            color = (base_color[0] * depth_factor, base_color[1] * depth_factor, base_color[2] * depth_factor)
            
            # Create marker based on class
            if class_name in ['car', 'bus', 'truck']:
                marker = self.create_vehicle_marker(prim_path, world_pos, color, angle_deg, relative_depth)
            elif class_name == 'person':
                marker = self.create_person_marker(prim_path, world_pos, color)
            else:
                marker = self.create_generic_marker(prim_path, world_pos, color)
            
            # Store prim path for updates
            self.detection_prims[track_id] = prim_path
            
            return marker
            
        except Exception as e:
            logger.error(f"Failed to create detection marker: {e}")
            return None
    
    def create_vehicle_marker(self, path, position, color, angle_deg, relative_depth=0.5):
        """Create a vehicle marker (box with orientation)"""
        try:
            vehicle = UsdGeom.Cube.Define(self.stage, path)
            vehicle.CreateSizeAttr(1.0)  # Fixed size - scaling handled by transform
            
            # Set transform
            xform = UsdGeom.Xformable(vehicle)
            xform.AddTranslateOp().Set(position)
            
            # Adjust scale based on depth (closer objects appear larger)
            base_scale = (8, 3, 4)  # Vehicle proportions (length, height, width)
            depth_scale_factor = 1.0 + (1.0 - relative_depth) * 0.5  # Range: 1.0 to 1.5
            scaled_dimensions = (base_scale[0] * depth_scale_factor, 
                               base_scale[1] * depth_scale_factor, 
                               base_scale[2] * depth_scale_factor)
            xform.AddScaleOp().Set(scaled_dimensions)
            
            # Apply rotation - convert angle to Y-axis rotation
            # Note: angle_deg is the horizontal angle from camera center
            xform.AddRotateYOp().Set(angle_deg)
            
            # Set color
            vehicle.CreateDisplayColorAttr([color])
            
            # Add label with angle and depth info
            self.create_label(f"{path}_label", position, f"Vehicle ({angle_deg:.1f}°, d:{relative_depth:.2f})")
            
            return vehicle
            
        except Exception as e:
            logger.error(f"Failed to create vehicle marker: {e}")
            return None
    
    def create_person_marker(self, path, position, color):
        """Create a person marker (cylinder)"""
        try:
            person = UsdGeom.Cylinder.Define(self.stage, path)
            person.CreateRadiusAttr(1.0)
            person.CreateHeightAttr(6.0)
            
            # Set transform
            xform = UsdGeom.Xformable(person)
            xform.AddTranslateOp().Set(position)
            
            # Set color
            person.CreateDisplayColorAttr([color])
            
            # Add label
            self.create_label(f"{path}_label", position, "Person")
            
            return person
            
        except Exception as e:
            logger.error(f"Failed to create person marker: {e}")
            return None
    
    def create_generic_marker(self, path, position, color):
        """Create a generic marker (sphere)"""
        try:
            marker = UsdGeom.Sphere.Define(self.stage, path)
            marker.CreateRadiusAttr(2.0)
            
            # Set transform
            xform = UsdGeom.Xformable(marker)
            xform.AddTranslateOp().Set(position)
            
            # Set color
            marker.CreateDisplayColorAttr([color])
            
            return marker
            
        except Exception as e:
            logger.error(f"Failed to create generic marker: {e}")
            return None
    
    def create_label(self, path, position, text):
        """Create a text label above the marker"""
        try:
            # Note: Text rendering in USD is complex, using a simple cube as placeholder
            label = UsdGeom.Cube.Define(self.stage, path)
            label.CreateSizeAttr(500.0)
            
            # Position above the marker
            label_pos = (position[0], position[1] + 8, position[2])
            xform = UsdGeom.Xformable(label)
            xform.AddTranslateOp().Set(label_pos)
            
            # White color for label
            label.CreateDisplayColorAttr([(1.0, 1.0, 1.0)])
            
        except Exception as e:
            logger.error(f"Failed to create label: {e}")
    
    def update_detection_marker(self, track_id, class_name, bbox, angle_deg, relative_depth=0.5):
        """Update existing detection marker"""
        try:
            prim_path = self.detection_prims.get(track_id)
            if not prim_path:
                return self.create_detection_marker(track_id, class_name, bbox, angle_deg, relative_depth)
            
            # Get the prim
            prim = self.stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                return self.create_detection_marker(track_id, class_name, bbox, angle_deg, relative_depth)
            
            # Update position
            world_pos = self.screen_to_world_coordinates(bbox, angle_deg)
            xform = UsdGeom.Xformable(prim)
            
            # Update translate operation
            translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
            if translate_ops:
                translate_ops[0].Set(world_pos)
            
            # Update rotation for vehicles using the calculated angle
            if class_name in ['car', 'bus', 'truck']:
                rotate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeRotateY]
                if rotate_ops:
                    rotate_ops[0].Set(angle_deg)
                    
            # Update label with new angle information
            label_path = f"{prim_path}_label"
            label_prim = self.stage.GetPrimAtPath(label_path)
            if label_prim.IsValid():
                # Update label text (this is a simplified approach)
                if class_name in ['car', 'bus', 'truck']:
                    # You could update label text here if needed
                    pass
            
            return prim
            
        except Exception as e:
            logger.error(f"Failed to update detection marker: {e}")
            return None
    
    def remove_old_detections(self, current_track_ids):
        """Remove markers for detections that are no longer present"""
        try:
            to_remove = []
            for track_id, prim_path in self.detection_prims.items():
                if track_id not in current_track_ids:
                    # Remove from stage
                    prim = self.stage.GetPrimAtPath(prim_path)
                    if prim.IsValid():
                        self.stage.RemovePrim(prim_path)
                    
                    # Remove label if exists
                    label_path = f"{prim_path}_label"
                    label_prim = self.stage.GetPrimAtPath(label_path)
                    if label_prim.IsValid():
                        self.stage.RemovePrim(label_path)
                    
                    to_remove.append(track_id)
            
            # Clean up tracking dictionary
            for track_id in to_remove:
                del self.detection_prims[track_id]
                
        except Exception as e:
            logger.error(f"Failed to remove old detections: {e}")
    
    async def fetch_detections(self):
        """Fetch detections from the API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/read", timeout=1.0) as response:
                    if response.status == 200:
                        detections = await response.json()
                        return detections if isinstance(detections, list) else []
                    else:
                        logger.warning(f"API returned status {response.status}")
                        return []
        except Exception as e:
            logger.debug(f"Failed to fetch detections: {e}")
            return []
    
    async def update_detections(self):
        """Update detection markers from API data"""
        try:
            detections = await self.fetch_detections()
            
            if not detections:
                return
            
            current_track_ids = set()
            
            # Process each detection
            for detection in detections:
                try:
                    track_id = detection.get('id')
                    class_name = detection.get('class', 'unknown')
                    bbox = detection.get('bbox', [0, 0, 100, 100])
                    angle_deg = detection.get('angle_deg', 0.0)
                    relative_depth = detection.get('relative_depth', 0.5)
                    
                    if track_id is not None:
                        current_track_ids.add(track_id)
                        self.update_detection_marker(track_id, class_name, bbox, angle_deg, relative_depth)
                        
                except Exception as e:
                    logger.error(f"Failed to process detection: {e}")
            
            # Remove old detections
            self.remove_old_detections(current_track_ids)
            
        except Exception as e:
            logger.error(f"Failed to update detections: {e}")

class OmniverseDetectionApp:
    """Main Omniverse application for detection visualization"""
    
    def __init__(self):
        self.visualizer = DetectionVisualizer()
        self.running = True
        self.viewport = None
        self.api_response = None
    
    def setup_viewport(self):
        """Setup the default viewport for detection visualization"""
        try:
            # Get the default viewport instead of creating a new one
            import omni.kit.viewport.utility as viewport_utils
            
            # Get the default viewport window
            viewport_windows = viewport_utils.get_active_viewport_window()
            if viewport_windows:
                self.viewport = viewport_windows # Use the first (default) viewport
                logger.info("Using default viewport for detection visualization")
            else:
                # Fallback: create a new viewport if none exists
                self.viewport = ViewportWindow("Detection Viewer", width=1280, height=720)
                self.viewport.visible = True
                logger.info("Created new viewport for detection visualization")
            
            # Set renderer to RTX
            act = actions.get_action_registry().get_action(
                "omni.kit.viewport.actions", "set_renderer_rtx_realtime"
            )
            if act:
                act.execute()
                logger.info("Set renderer to RTX Realtime")
            else:
                logger.warning("RTX action not found - is omni.kit.viewport.actions enabled?")
            
            return True
        except Exception as e:
            logger.error(f"Failed to setup viewport: {e}")
            # Fallback to creating new viewport
            try:
                self.viewport = ViewportWindow("Detection Viewer", width=1280, height=720)
                self.viewport.visible = True
                return True
            except:
                return False
        
    def create_ui(self):
        """Create UI window"""
        try:
            self.window = ui.Window("Detection Visualizer", width=400, height=400)
            with self.window.frame:
                with ui.VStack(spacing=10):
                    ui.Label("Real-Time Detection Visualization", height=30)
                    
                    # API Status Section
                    with ui.HStack():
                        ui.Label("API Status:", width=100)
                        self.status_label = ui.Label("Connecting...", style={"color": 0xFFFF0000})
                    
                    with ui.HStack():
                        ui.Label("Detections:", width=100)
                        self.detection_count = ui.Label("0", style={"color": 0xFF00FF00})
                    
                    with ui.HStack():
                        ui.Label("Update Rate:", width=100)
                        self.update_rate = ui.Label("0.0 Hz", style={"color": 0xFF00FFFF})
                    
                    ui.Separator()
                    
                    # API Control Section
                    ui.Label("API Controls:", height=20)
                    
                    with ui.HStack(spacing=5):
                        start_btn = ui.Button("Start Detection", height=30, width=120)
                        start_btn.set_clicked_fn(self.start_detection)
                        
                        stop_btn = ui.Button("Stop Detection", height=30, width=120)
                        stop_btn.set_clicked_fn(self.stop_detection)
                    
                    with ui.HStack(spacing=5):
                        read_btn = ui.Button("Read Detections", height=30, width=120)
                        read_btn.set_clicked_fn(self.read_detections)
                        
                        test_btn = ui.Button("Test API", height=30, width=120)
                        test_btn.set_clicked_fn(self.test_api)
                    
                    # API Response Display
                    ui.Label("API Response:", height=20)
                    self.api_response = ui.Label("Ready", word_wrap=True, style={"color": 0xFFFFFFFF})
                    
                    ui.Separator()
                    
                    # Info Section
                    ui.Label("Coordinate Mapping:", height=20)
                    ui.Label(f"• FOV: {config.fov_deg}° for perspective projection")
                    ui.Label(f"• Image: {self.visualizer.image_width}x{self.visualizer.image_height}")
                    ui.Label(f"• Depth range: {self.visualizer.min_distance}-{self.visualizer.max_distance} units")
                    
                    # Coordinate tuning controls
                    ui.Label("Coordinate Tuning:", height=20)
                    with ui.HStack():
                        ui.Label("Ref Distance:", width=80)
                        self.ref_distance_field = ui.IntField(width=60)
                        self.ref_distance_field.model.set_value(int(self.visualizer.reference_distance))
                        self.ref_distance_field.model.add_value_changed_fn(self.on_ref_distance_changed)
                    
                    with ui.HStack():
                        ui.Label("Ref Height:", width=80)
                        self.ref_height_field = ui.IntField(width=60)
                        self.ref_height_field.model.set_value(int(self.visualizer.reference_bbox_height))
                        self.ref_height_field.model.add_value_changed_fn(self.on_ref_height_changed)
                    
                    with ui.HStack():
                        ui.Label("Min Distance:", width=80)
                        self.min_dist_field = ui.IntField(width=60)
                        self.min_dist_field.model.set_value(int(self.visualizer.min_distance))
                        self.min_dist_field.model.add_value_changed_fn(self.on_min_distance_changed)
                    
                    with ui.HStack():
                        ui.Label("Max Distance:", width=80)
                        self.max_dist_field = ui.IntField(width=60)
                        self.max_dist_field.model.set_value(int(self.visualizer.max_distance))
                        self.max_dist_field.model.add_value_changed_fn(self.on_max_distance_changed)
                    
                    ui.Separator()
                    
                    # Scene Controls
                    reset_btn = ui.Button("Reset Scene", height=30)
                    reset_btn.set_clicked_fn(self.reset_scene)
                        
        except Exception as e:
            logger.error(f"Failed to create UI: {e}")
    
    def start_detection(self):
        """Start detection via API"""
        logger.info("=== START DETECTION BUTTON CLICKED ===")
        async def _start():
            try:
                logger.info(f"Starting detection via {self.visualizer.api_url}/start")
                timeout = aiohttp.ClientTimeout(total=5.0)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(f"{self.visualizer.api_url}/start") as response:
                        logger.info(f"Start response status: {response.status}")
                        if response.status == 200:
                            result = await response.json()
                            message = f"Start: {result.get('message', 'Success')}"
                            if hasattr(self, 'api_response') and self.api_response:
                                self.api_response.text = message
                            logger.info("Detection started successfully")
                        else:
                            message = f"Start failed: {response.status}"
                            if hasattr(self, 'api_response') and self.api_response:
                                self.api_response.text = message
                            logger.error(f"Failed to start detection: {response.status}")
            except aiohttp.ClientConnectorError as e:
                message = f"Start: Connection failed - {str(e)}"
                if hasattr(self, 'api_response') and self.api_response:
                    self.api_response.text = message
                logger.error(f"Start connection error: {e}")
            except asyncio.TimeoutError as e:
                message = f"Start: Timeout - {str(e)}"
                if hasattr(self, 'api_response') and self.api_response:
                    self.api_response.text = message
                logger.error(f"Start timeout error: {e}")
            except Exception as e:
                message = f"Start error: {str(e)} ({type(e).__name__})"
                if hasattr(self, 'api_response') and self.api_response:
                    self.api_response.text = message
                logger.error(f"Error starting detection: {e} (type: {type(e).__name__})")
        
        # Run the async function
        asyncio.ensure_future(_start())
    
    def stop_detection(self):
        """Stop detection via API"""
        async def _stop():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{self.visualizer.api_url}/stop", timeout=5.0) as response:
                        if response.status == 200:
                            result = await response.json()
                            message = f"Stop: {result.get('message', 'Success')}"
                            if hasattr(self, 'api_response') and self.api_response:
                                self.api_response.text = message
                            logger.info("Detection stopped successfully")
                        else:
                            message = f"Stop failed: {response.status}"
                            if hasattr(self, 'api_response') and self.api_response:
                                self.api_response.text = message
                            logger.error(f"Failed to stop detection: {response.status}")
            except Exception as e:
                message = f"Stop error: {str(e)}"
                if hasattr(self, 'api_response') and self.api_response:
                    self.api_response.text = message
                logger.error(f"Error stopping detection: {e}")
        
        # Run the async function
        asyncio.ensure_future(_stop())
    
    def read_detections(self):
        """Read current detections via API"""
        async def _read():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.visualizer.api_url}/read", timeout=5.0) as response:
                        if response.status == 200:
                            detections = await response.json()
                            count = len(detections) if isinstance(detections, list) else 0
                            message = f"Read: Found {count} detections"
                            if hasattr(self, 'api_response') and self.api_response:
                                self.api_response.text = message
                            logger.info(f"Read {count} detections successfully")
                        else:
                            message = f"Read failed: {response.status}"
                            if hasattr(self, 'api_response') and self.api_response:
                                self.api_response.text = message
                            logger.error(f"Failed to read detections: {response.status}")
            except Exception as e:
                message = f"Read error: {str(e)}"
                if hasattr(self, 'api_response') and self.api_response:
                    self.api_response.text = message
                logger.error(f"Error reading detections: {e}")
        
        # Run the async function
        asyncio.ensure_future(_read())
    
    def test_api(self):
        """Test API health endpoint"""
        logger.info("=== TEST API BUTTON CLICKED ===")
        async def _test():
            try:
                logger.info(f"Testing API connection to {self.visualizer.api_url}/health")
                timeout = aiohttp.ClientTimeout(total=5.0)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(f"{self.visualizer.api_url}/health") as response:
                        logger.info(f"API response status: {response.status}")
                        if response.status == 200:
                            message = "API Test: Connected ✓"
                            if hasattr(self, 'api_response') and self.api_response:
                                self.api_response.text = message
                            logger.info("API health check successful")
                        else:
                            message = f"API Test: Failed ({response.status})"
                            if hasattr(self, 'api_response') and self.api_response:
                                self.api_response.text = message
                            logger.error(f"API health check failed: {response.status}")
            except aiohttp.ClientConnectorError as e:
                message = f"API Test: Connection failed - {str(e)}"
                if hasattr(self, 'api_response') and self.api_response:
                    self.api_response.text = message
                logger.error(f"API connection error: {e}")
            except asyncio.TimeoutError as e:
                message = f"API Test: Timeout - {str(e)}"
                if hasattr(self, 'api_response') and self.api_response:
                    self.api_response.text = message
                logger.error(f"API timeout error: {e}")
            except Exception as e:
                message = f"API Test: Error - {str(e)} ({type(e).__name__})"
                if hasattr(self, 'api_response') and self.api_response:
                    self.api_response.text = message
                logger.error(f"API health check error: {e} (type: {type(e).__name__})")
        
        # Run the async function
        asyncio.ensure_future(_test())
    
    def on_ref_distance_changed(self, model):
        """Update reference distance for coordinate conversion"""
        try:
            new_value = model.get_value_as_int()
            self.visualizer.reference_distance = max(1, new_value)
            logger.info(f"Reference distance updated to: {self.visualizer.reference_distance}")
        except Exception as e:
            logger.error(f"Error updating reference distance: {e}")
    
    def on_ref_height_changed(self, model):
        """Update reference height for coordinate conversion"""
        try:
            new_value = model.get_value_as_int()
            self.visualizer.reference_bbox_height = max(1, new_value)
            logger.info(f"Reference bbox height updated to: {self.visualizer.reference_bbox_height}")
        except Exception as e:
            logger.error(f"Error updating reference height: {e}")
    
    def on_min_distance_changed(self, model):
        """Update minimum distance for coordinate conversion"""
        try:
            new_value = model.get_value_as_int()
            self.visualizer.min_distance = max(1, new_value)
            logger.info(f"Minimum distance updated to: {self.visualizer.min_distance}")
        except Exception as e:
            logger.error(f"Error updating minimum distance: {e}")
    
    def on_max_distance_changed(self, model):
        """Update maximum distance for coordinate conversion"""
        try:
            new_value = model.get_value_as_int()
            self.visualizer.max_distance = max(self.visualizer.min_distance + 1, new_value)
            logger.info(f"Maximum distance updated to: {self.visualizer.max_distance}")
        except Exception as e:
            logger.error(f"Error updating maximum distance: {e}")
    
    def frame_stage(self):
        """Frame the stage content in viewport"""
        try:
            if not self.viewport or not self.visualizer.stage:
                return
            
            # Get default prim or frame root
            dp = self.visualizer.stage.GetDefaultPrim()
            targets = [dp.GetPath().pathString] if dp else ["/"]
            
            # Frame the content - try different command formats
            try:
                # Try with viewport parameter (newer versions)
                omni.kit.commands.execute(
                    "FramePrimsCommand",
                    viewport=self.viewport.viewport_api,
                    prim_paths=targets,
                )
            except (TypeError, AttributeError):
                try:
                    # Try with paths parameter (older versions)
                    omni.kit.commands.execute(
                        "FramePrimsCommand",
                        paths=targets,
                    )
                except (TypeError, AttributeError):
                    # Fallback - try basic framing
                    logger.warning("Could not frame stage - FramePrimsCommand not available")
            logger.info(f"Framed: {targets}")
            
        except Exception as e:
            logger.error(f"Failed to frame stage: {e}")
    
    def reset_scene(self):
        """Reset the 3D scene"""
        try:
            # Clear all detection markers
            if self.visualizer.stage:
                detection_root = self.visualizer.stage.GetPrimAtPath("/Detections")
                if detection_root.IsValid():
                    self.visualizer.stage.RemovePrim("/Detections")
                
                # Recreate detection root
                self.visualizer.detection_root = UsdGeom.Xform.Define(self.visualizer.stage, "/Detections")
                self.visualizer.detection_prims.clear()
                
            logger.info("Scene reset complete")
            
        except Exception as e:
            logger.error(f"Failed to reset scene: {e}")
    
    async def update_loop(self):
        """Main update loop"""
        last_update = 0
        update_count = 0
        
        while self.running:
            try:
                # Check if Kit app is still running
                app = omni.kit.app.get_app()
                if app and app.is_running():
                    current_time = time.time()
                    
                    if current_time - last_update >= self.visualizer.update_interval:
                        await self.visualizer.update_detections()
                        last_update = current_time
                        update_count += 1
                        
                        # Update UI every 10 updates
                        if update_count % 10 == 0:
                            self.update_ui_stats(current_time)
                    
                    await asyncio.sleep(0.01)  # Small sleep to prevent CPU spinning
                else:
                    # Kit app is shutting down
                    self.running = False
                    break
                
            except asyncio.CancelledError:
                logger.info("Update loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(1.0)
    
    def update_ui_stats(self, current_time):
        """Update UI statistics"""
        try:
            detection_count = len(self.visualizer.detection_prims)
            update_rate = 1.0 / self.visualizer.update_interval
            
            if hasattr(self, 'detection_count'):
                self.detection_count.text = str(detection_count)
            if hasattr(self, 'update_rate'):
                self.update_rate.text = f"{update_rate:.1f} Hz"
            if hasattr(self, 'status_label'):
                self.status_label.text = "Connected" if detection_count > 0 else "No Data"
                
        except Exception as e:
            logger.error(f"Failed to update UI stats: {e}")

def main():
    """Main entry point when executed via kit.exe --exec"""
    try:
        logger.info("=== Omniverse Detection Visualizer Starting ===")
        
        # Get USD file path from command line arguments if provided
        usd_arg = sys.argv[1] if len(sys.argv) > 1 else ""
        
        # Get the Kit app instance
        app = omni.kit.app.get_app()
        if not app:
            logger.error("No Kit application instance found")
            return
        
        # Create application instance
        global app_instance
        app_instance = OmniverseDetectionApp()
        
        # # Setup viewport first
        # if not app_instance.setup_viewport():
        #     logger.error("Failed to setup viewport")
        #     return
        
        # Setup stage with USD file
        if not app_instance.visualizer.setup_stage(usd_arg):
            logger.error("Failed to setup USD stage")
            return
        
        # Subscribe to stage events to frame after load
        def _on_stage_loaded(e):
            from omni.usd import StageEventType
            if e.type == StageEventType.OPENED:
                # Frame the stage content after it loads
                app_instance.frame_stage()
        
        ctx = omni.usd.get_context()
        ev = ctx.get_stage_event_stream()
        _stage_sub = ev.create_subscription_to_pop(_on_stage_loaded)
        
        # Frame immediately if stage is already loaded
        if app_instance.visualizer.stage:
            app_instance.frame_stage()
        
        # Create UI
        app_instance.create_ui()
        
        # Start the async update loop
        async def run_update_loop():
            try:
                await app_instance.update_loop()
            except Exception as e:
                logger.error(f"Update loop error: {e}")
        
        # Run the async task
        task = asyncio.ensure_future(run_update_loop())
        
        # Register shutdown handler
        def on_shutdown(_):
            logger.info("Shutting down Detection Visualizer")
            app_instance.running = False
            if task and not task.done():
                task.cancel()
        
        # Subscribe to shutdown event
        shutdown_stream = app.get_shutdown_event_stream()
        shutdown_sub = shutdown_stream.create_subscription_to_pop(on_shutdown, name="detection_viz_shutdown")
        
        # Keep subscriptions alive
        global _shutdown_sub, _stage_subscription
        _shutdown_sub = shutdown_sub
        _stage_subscription = _stage_sub
        
        logger.info("Detection Visualizer initialized successfully")
        logger.info("Waiting for detections from API at http://127.0.0.1:8000")
        logger.info("IMPORTANT: Make sure the API server is running!")
        logger.info("To start the API server, run: python API/app.py")
        logger.info("Or use the batch file: run_local_demo.bat")
        logger.info("Use the 'Test API' button in the UI to check connection")
        
    except Exception as e:
        logger.error(f"Failed to initialize Detection Visualizer: {e}")
        import traceback
        traceback.print_exc()

# Execute main when script is loaded by kit.exe
main()