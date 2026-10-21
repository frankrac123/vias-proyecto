"""
Configuration management for Traffic Detection System
Handles loading, saving, and managing application configuration
"""
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple
from datetime import datetime


@dataclass
class LineConfig:
    """Configuration for counting line"""
    x1: float = 0.5
    y1: float = 0.0
    x2: float = 0.5
    y2: float = 1.0
    name: str = "Línea Principal"
    enabled: bool = True

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(data):
        return LineConfig(**data)


@dataclass
class DetectionConfig:
    """Configuration for YOLO detection"""
    model_path: str = "yolo11n.pt"
    confidence: float = 0.30
    imgsz: int = 416
    device: str = "auto"  # auto, cpu, cuda
    classes: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 5, 7])


@dataclass
class CaptureConfig:
    """Configuration for video capture"""
    source: str = "0"
    source_type: str = "usb"  # mjpeg, rtsp, file, usb
    frame_buffer_size: int = 2
    reconnect_attempts: int = 5
    reconnect_delay: float = 2.0


@dataclass
class TrackerConfig:
    """Configuration for object tracking"""
    tracker_type: str = "bytetrack"  # bytetrack, botsort
    max_age: int = 30
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.05


@dataclass
class AnalyticsConfig:
    """Configuration for analytics"""
    intervals: List[int] = field(default_factory=lambda: [1, 5, 15, 60])  # minutes
    store_trajectory: bool = True
    trajectory_max_length: int = 50
    calculate_speed: bool = False


@dataclass
class DisplayConfig:
    """Configuration for GUI display"""
    show_confidence: bool = True
    show_track_id: bool = True
    show_trajectory: bool = True
    trajectory_length: int = 20
    confidence_threshold_display: float = 0.30


@dataclass
class ApplicationConfig:
    """Main application configuration"""
    camera: CaptureConfig = field(default_factory=CaptureConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    line: LineConfig = field(default_factory=LineConfig)
    
    export_path: str = "./exports"
    data_path: str = "./data"
    sessions_path: str = "./sessions"
    
    # Performance modes: quality, balanced, performance
    performance_mode: str = "balanced"
    
    # GUI settings
    window_width: int = 1400
    window_height: int = 900
    dark_theme: bool = True


class ConfigManager:
    """Manages loading and saving configuration"""
    
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "config.json"
        self.config = self._load_config()
        self._apply_performance_mode()
    
    def _load_config(self) -> ApplicationConfig:
        """Load configuration from file or create default"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Reconstruct nested dataclass objects
                camera = CaptureConfig(**data.get('camera', {}))
                detection = DetectionConfig(**data.get('detection', {}))
                tracker = TrackerConfig(**data.get('tracker', {}))
                analytics = AnalyticsConfig(**data.get('analytics', {}))
                display = DisplayConfig(**data.get('display', {}))
                line = LineConfig(**data.get('line', {}))
                
                config = ApplicationConfig(
                    camera=camera,
                    detection=detection,
                    tracker=tracker,
                    analytics=analytics,
                    display=display,
                    line=line,
                    export_path=data.get('export_path', './exports'),
                    data_path=data.get('data_path', './data'),
                    sessions_path=data.get('sessions_path', './sessions'),
                    performance_mode=data.get('performance_mode', 'balanced'),
                    window_width=data.get('window_width', 1400),
                    window_height=data.get('window_height', 900),
                    dark_theme=data.get('dark_theme', True)
                )
                return config
            except Exception as e:
                print(f"Error loading config: {e}. Using defaults.")
                return ApplicationConfig()
        else:
            return ApplicationConfig()
    
    def _apply_performance_mode(self):
        """Apply performance mode settings"""
        mode = self.config.performance_mode.lower()
        
        if mode == "quality":
            self.config.detection.imgsz = 640
        elif mode == "performance":
            self.config.detection.imgsz = 320
        else:  # balanced
            self.config.detection.imgsz = 416
    
    def save(self):
        """Save current configuration to file"""
        config_dict = {
            'camera': asdict(self.config.camera),
            'detection': asdict(self.config.detection),
            'tracker': asdict(self.config.tracker),
            'analytics': asdict(self.config.analytics),
            'display': asdict(self.config.display),
            'line': asdict(self.config.line),
            'export_path': self.config.export_path,
            'data_path': self.config.data_path,
            'sessions_path': self.config.sessions_path,
            'performance_mode': self.config.performance_mode,
            'window_width': self.config.window_width,
            'window_height': self.config.window_height,
            'dark_theme': self.config.dark_theme,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def update_camera_source(self, source: str, source_type: str = "mjpeg"):
        """Update camera source"""
        self.config.camera.source = source
        self.config.camera.source_type = source_type
        self.save()
    
    def update_line(self, x1: float, y1: float, x2: float, y2: float, name: str = "Línea Principal"):
        """Update counting line coordinates (normalized)"""
        self.config.line.x1 = x1
        self.config.line.y1 = y1
        self.config.line.x2 = x2
        self.config.line.y2 = y2
        self.config.line.name = name
        self.save()
    
    def get_config(self) -> ApplicationConfig:
        """Get current configuration"""
        return self.config
