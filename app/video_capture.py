"""
Improved asynchronous video capture module with multi-backend support
Supports MJPEG (HTTP), RTSP, USB cameras with low-latency streaming
"""
import cv2
import threading
import time
import numpy as np
from typing import Optional, Tuple, Dict
from collections import deque
import warnings
import logging
import os

try:
    import av
    HAS_PYAV = True
except ImportError:
    av = None
    HAS_PYAV = False

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Suppress OpenCV warnings
warnings.filterwarnings('ignore', category=Warning)
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_FFMPEG_DEBUG'] = '0'


class AsyncVideoCapture:
    """
    Thread-safe video capture with minimal latency.
    
    Features:
    - Only keeps the most recent frame (buffer size = 1)
    - Automatic reconnection on connection loss
    - Multi-backend support (MJPEG, RTSP, USB, file)
    - Low-latency streaming options
    - Thread-safe frame access
    - FPS calculation and monitoring
    """
    
    def __init__(self, source: str, source_type: str = "auto", 
                 max_buffer_size: int = 1,
                 reconnect_attempts: int = 10,
                 reconnect_delay: float = 1.0,
                 timeout: int = 10):
        """
        Initialize async video capture.
        
        Args:
            source: Video source (URL, file path, or device index)
            source_type: Type of source ('mjpeg', 'rtsp', 'http', 'file', 'usb', 'auto')
            max_buffer_size: Maximum frames to buffer (keep at 1 for low latency)
            reconnect_attempts: Number of reconnection attempts before giving up
            reconnect_delay: Delay between reconnection attempts (seconds)
            timeout: Connection timeout in seconds
        """
        self.source = source
        self.source_type = source_type
        self.max_buffer_size = max(1, max_buffer_size)
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.timeout = timeout
        
        # Frame buffer and locking
        self.frame_buffer = deque(maxlen=self.max_buffer_size)
        self.frame_lock = threading.Lock()
        self.last_frame_time = time.time()
        
        # Capture state
        self.cap = None
        self.av_container = None
        self.av_frames = None
        self.backend = None
        self.is_running = False
        self.is_connected = False
        self.capture_thread = None
        
        # Statistics
        self.frame_count = 0
        # Monotonic sequence used by consumers to tell whether a frame is new.
        # frame_count is only the one-second FPS window and is reset periodically.
        self.total_frames_captured = 0
        self.fps = 0.0
        self.last_fps_update = time.time()
        self.frames_dropped = 0
        self.connection_attempts = 0
        self.connection_error: Optional[str] = None
        
        # Video properties
        self.frame_width = 0
        self.frame_height = 0
        self.frame_fps = 0.0
    
    def _get_source_url(self) -> str:
        """Construct proper URL for OpenCV depending on source type."""
        if self.source_type == "rtsp":
            return self.source
        elif self.source_type in ["mjpeg", "http"]:
            # Ensure HTTP URLs are properly formatted
            if not self.source.startswith(("http://", "https://")):
                return f"http://{self.source}"
            return self.source
        elif self.source_type == "file":
            return self.source
        elif self.source_type == "usb":
            # Try to parse as integer (device index)
            try:
                return int(self.source)
            except ValueError:
                return self.source
        else:  # auto-detect
            if self.source.startswith(("http://", "https://")):
                return self.source
            elif self.source.startswith("rtsp://"):
                return self.source
            else:
                try:
                    return int(self.source)
                except ValueError:
                    return self.source
    
    def _create_capture(self) -> bool:
        """Create video capture object with optimized settings for source type."""
        try:
            self.connection_attempts += 1
            url = self._get_source_url()
            
            logger.info(f"Attempting connection (attempt {self.connection_attempts}): {url}")

            if (HAS_PYAV and not isinstance(url, int) and
                    self.source_type in ("mjpeg", "http", "rtsp", "auto")):
                try:
                    options = {
                        "fflags": "nobuffer",
                        "flags": "low_delay",
                        "probesize": "32768",
                        "analyzeduration": "0",
                        "rw_timeout": str(self.timeout * 1_000_000),
                    }
                    if self.source_type == "rtsp":
                        options["rtsp_transport"] = "tcp"
                    self.av_container = av.open(
                        url, mode="r", options=options, timeout=self.timeout
                    )
                    video_stream = next(
                        stream for stream in self.av_container.streams
                        if stream.type == "video"
                    )
                    video_stream.thread_type = "AUTO"
                    self.av_frames = self.av_container.decode(video=video_stream.index)
                    frame = next(self.av_frames).to_ndarray(format="bgr24")
                    self.frame_height, self.frame_width = frame.shape[:2]
                    self.frame_fps = float(video_stream.average_rate or 0)
                    with self.frame_lock:
                        self.frame_buffer.append(frame)
                        self.last_frame_time = time.time()
                    self.backend = "pyav"
                    self.is_connected = True
                    self.connection_error = None
                    return True
                except Exception as exc:
                    logger.warning(f"PyAV failed; using OpenCV fallback: {exc}")
                    if self.av_container:
                        self.av_container.close()
                    self.av_container = None
                    self.av_frames = None
            
            # Create capture with appropriate backend
            if isinstance(url, int):
                # USB camera
                self.cap = cv2.VideoCapture(url)
            else:
                # Network source - use CAP_FFMPEG backend explicitly
                self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            
            if not self.cap.isOpened():
                logger.warning(f"Failed to open capture source: {url}")
                self.connection_error = "Failed to open capture source"
                return False
            
            # Set buffer size to 1 to prevent frame accumulation
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set timeouts and low-delay options for MJPEG/RTSP
            if self.source_type in ["mjpeg", "rtsp", "http"]:
                try:
                    # Try to set stream timeout (backend-specific)
                    self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                    
                    # Disable hwaccel if available (can cause lag)
                    # self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)
                except:
                    pass
            
            # Try to read first frame to verify connection
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning("Failed to read first frame from source")
                self.cap.release()
                self.cap = None
                self.connection_error = "Failed to read first frame"
                return False
            
            # Get video properties
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Connected! Resolution: {self.frame_width}x{self.frame_height}, FPS: {self.frame_fps:.1f}")
            
            # Add first frame to buffer
            with self.frame_lock:
                self.frame_buffer.append(frame)
                self.last_frame_time = time.time()
            
            self.is_connected = True
            self.backend = "opencv"
            self.connection_error = None
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.connection_error = str(e)
            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass
            self.cap = None
            return False
    
    def start(self) -> bool:
        """Start the capture thread."""
        if self.is_running:
            logger.warning("Capture already running")
            return True
        
        self.is_running = True
        self.frame_count = 0
        self.total_frames_captured = 0
        self.frames_dropped = 0
        self.connection_attempts = 0
        
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="AsyncVideoCapture"
        )
        self.capture_thread.start()
        # Connection is intentionally asynchronous. Waiting here would block the
        # Qt main thread for every FFmpeg timeout when a camera is unavailable.
        return True
    
    def stop(self):
        """Stop capture thread and cleanup."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
        
        self.cap = None
        if self.av_container:
            try:
                self.av_container.close()
            except Exception:
                pass
        self.av_container = None
        self.av_frames = None
        self.is_connected = False
        logger.info("Capture stopped")
    
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        reconnect_count = 0
        consecutive_errors = 0
        
        while self.is_running:
            # Try to connect if disconnected
            if not self.is_connected:
                if reconnect_count >= self.reconnect_attempts:
                    # Reset counter and wait before trying again
                    time.sleep(self.reconnect_delay * 2)
                    reconnect_count = 0
                    consecutive_errors = 0
                
                if self._create_capture():
                    reconnect_count = 0
                    consecutive_errors = 0
                else:
                    reconnect_count += 1
                    consecutive_errors += 1
                    time.sleep(min(self.reconnect_delay * consecutive_errors, 5.0))
                    continue
            
            # Capture frames
            try:
                if self.backend == "pyav":
                    frame = next(self.av_frames).to_ndarray(format="bgr24")
                    ret = True
                else:
                    ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    # Store frame in buffer (replaces old frame)
                    with self.frame_lock:
                        self.frame_buffer.append(frame)
                        self.last_frame_time = time.time()
                    
                    # Update FPS
                    self.frame_count += 1
                    self.total_frames_captured += 1
                    current_time = time.time()
                    if current_time - self.last_fps_update >= 1.0:
                        self.fps = self.frame_count / (current_time - self.last_fps_update)
                        self.frame_count = 0
                        self.last_fps_update = current_time
                    
                    consecutive_errors = 0
                    
                else:
                    # Frame read failed
                    logger.warning(f"Failed to read frame (ret={ret})")
                    self.is_connected = False
                    consecutive_errors += 1
                    
                    if self.cap:
                        try:
                            self.cap.release()
                        except:
                            pass
                    self.cap = None
                    if self.av_container:
                        self.av_container.close()
                    self.av_container = None
                    self.av_frames = None
                    
            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                self.connection_error = str(e)
                self.is_connected = False
                consecutive_errors += 1
                
                if self.cap:
                    try:
                        self.cap.release()
                    except:
                        pass
                self.cap = None
                if self.av_container:
                    try:
                        self.av_container.close()
                    except Exception:
                        pass
                self.av_container = None
                self.av_frames = None
                
                time.sleep(self.reconnect_delay)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the most recent frame from buffer.
        
        Returns:
            Tuple of (success, frame)
        """
        with self.frame_lock:
            if len(self.frame_buffer) > 0:
                frame = self.frame_buffer[-1].copy()  # Return copy to avoid external modifications
                return True, frame
        return False, None
    
    def get_fps(self) -> float:
        """Get current capture FPS."""
        return self.fps
    
    def get_frame_shape(self) -> Tuple[int, int, int]:
        """Get frame shape (height, width, channels)."""
        if self.is_connected:
            try:
                with self.frame_lock:
                    if len(self.frame_buffer) > 0:
                        return self.frame_buffer[-1].shape
            except:
                pass
        return (480, 640, 3)  # Default fallback
    
    def get_dimensions(self) -> Tuple[int, int]:
        """Get video dimensions (width, height)."""
        return self.frame_width, self.frame_height
    
    def is_available(self) -> bool:
        """Check if capture is running and connected."""
        return self.is_running and self.is_connected
    
    def get_status(self) -> Dict:
        """Get capture status information."""
        return {
            'connected': self.is_connected,
            'running': self.is_running,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'total_frames_captured': self.total_frames_captured,
            'frames_dropped': self.frames_dropped,
            'width': self.frame_width,
            'height': self.frame_height,
            'buffer_size': len(self.frame_buffer),
            'error': self.connection_error,
            'backend': self.backend,
            'connection_attempts': self.connection_attempts
        }
    
    def get_connection_status(self) -> Tuple[bool, Optional[str]]:
        """Get connection status with error message."""
        return self.is_connected, self.connection_error
