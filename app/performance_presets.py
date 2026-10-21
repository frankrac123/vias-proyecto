"""
Performance presets for traffic detection system
Allows users to choose between quality, balanced, and performance modes
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class PerformanceMode(Enum):
    """Available performance modes"""
    MAXIMUM_QUALITY = "MÁXIMA CALIDAD"
    BALANCED = "BALANCEADO"
    HIGH_PERFORMANCE = "ALTO RENDIMIENTO"


@dataclass
class PerformancePreset:
    """Performance preset configuration"""
    mode: PerformanceMode
    
    # YOLO settings
    yolo_model: str  # yolo11n, yolo11s, yolo11m
    yolo_imgsz: int  # Input size for YOLO
    yolo_confidence: float
    yolo_iou: float
    
    # Inference settings
    inference_fps_limit: Optional[int]  # Max inference FPS (None = unlimited)
    
    # Tracking settings
    max_tracks: int  # Max simultaneous tracks to maintain
    track_history_length: int
    
    # Display settings
    draw_trajectories: bool
    draw_bounding_boxes: bool
    draw_labels: bool
    
    # Capture settings
    buffer_size: int
    skip_frames: int  # Process every Nth frame
    
    def to_dict(self) -> Dict:
        """Convert to configuration dictionary"""
        return {
            'mode': self.mode.value,
            'yolo_model': self.yolo_model,
            'yolo_imgsz': self.yolo_imgsz,
            'yolo_confidence': self.yolo_confidence,
            'yolo_iou': self.yolo_iou,
            'inference_fps_limit': self.inference_fps_limit,
            'max_tracks': self.max_tracks,
            'track_history_length': self.track_history_length,
            'draw_trajectories': self.draw_trajectories,
            'draw_bounding_boxes': self.draw_bounding_boxes,
            'draw_labels': self.draw_labels,
            'buffer_size': self.buffer_size,
            'skip_frames': self.skip_frames,
        }


# Predefined presets
PRESETS = {
    PerformanceMode.MAXIMUM_QUALITY: PerformancePreset(
        mode=PerformanceMode.MAXIMUM_QUALITY,
        yolo_model='yolo11s',
        yolo_imgsz=640,
        yolo_confidence=0.50,
        yolo_iou=0.50,
        inference_fps_limit=None,  # No limit
        max_tracks=200,
        track_history_length=100,
        draw_trajectories=True,
        draw_bounding_boxes=True,
        draw_labels=True,
        buffer_size=1,
        skip_frames=0,
    ),
    
    PerformanceMode.BALANCED: PerformancePreset(
        mode=PerformanceMode.BALANCED,
        yolo_model='yolo11n',
        yolo_imgsz=416,
        yolo_confidence=0.35,
        yolo_iou=0.45,
        inference_fps_limit=None,
        max_tracks=150,
        track_history_length=50,
        draw_trajectories=True,
        draw_bounding_boxes=True,
        draw_labels=True,
        buffer_size=1,
        skip_frames=0,
    ),
    
    PerformanceMode.HIGH_PERFORMANCE: PerformancePreset(
        mode=PerformanceMode.HIGH_PERFORMANCE,
        yolo_model='yolo11n',
        yolo_imgsz=320,
        yolo_confidence=0.30,
        yolo_iou=0.40,
        inference_fps_limit=10,  # Limit to 10 FPS
        max_tracks=100,
        track_history_length=20,
        draw_trajectories=False,  # Minimal drawing
        draw_bounding_boxes=True,
        draw_labels=False,
        buffer_size=1,
        skip_frames=1,  # Process every 2nd frame
    ),
}


class PerformanceManager:
    """Manage performance presets"""
    
    def __init__(self):
        self.current_preset = PRESETS[PerformanceMode.BALANCED]
        self.custom_presets = {}
    
    def set_mode(self, mode: PerformanceMode):
        """Switch to a predefined mode"""
        if mode in PRESETS:
            self.current_preset = PRESETS[mode]
            return True
        return False
    
    def get_current_config(self) -> Dict:
        """Get current preset as configuration dict"""
        return self.current_preset.to_dict()
    
    def get_all_modes(self) -> list:
        """Get available modes"""
        return [mode.value for mode in PerformanceMode]
    
    def create_custom_preset(self, name: str, **kwargs) -> bool:
        """Create custom preset from current settings"""
        try:
            preset_dict = self.current_preset.to_dict()
            preset_dict.update(kwargs)
            self.custom_presets[name] = preset_dict
            return True
        except Exception as e:
            print(f"Error creating custom preset: {e}")
            return False
    
    def get_preset(self, mode_or_name: str):
        """Get preset by mode or custom name"""
        # Try predefined modes
        for mode in PerformanceMode:
            if mode.value == mode_or_name:
                return PRESETS[mode].to_dict()
        
        # Try custom presets
        if mode_or_name in self.custom_presets:
            return self.custom_presets[mode_or_name]
        
        return None
    
    def get_description(self, mode: PerformanceMode) -> str:
        """Get description of mode"""
        descriptions = {
            PerformanceMode.MAXIMUM_QUALITY: (
                "Máxima precisión de detección.\n"
                "Modelo YOLO11s, resolución 640p, sin límite de FPS.\n"
                "Ideal para análisis detallado pero más lento.\n"
                "Requiere: CPU rápida o GPU"
            ),
            PerformanceMode.BALANCED: (
                "Balance óptimo entre velocidad y precisión.\n"
                "Modelo YOLO11n, resolución 416p, procesamiento en tiempo real.\n"
                "Recomendado para la mayoría de casos.\n"
                "Requiere: CPU estándar"
            ),
            PerformanceMode.HIGH_PERFORMANCE: (
                "Prioriza baja latencia y rapidez.\n"
                "Modelo YOLO11n, resolución 320p, límite 10 FPS de inferencia.\n"
                "Para hardware limitado o múltiples cámaras.\n"
                "Sacrifica algo de precisión por velocidad.\n"
                "Requiere: Hardware básico"
            ),
        }
        return descriptions.get(mode, "")


# Quick access function
def get_preset_config(mode_name: str) -> Dict:
    """Quick preset access"""
    manager = PerformanceManager()
    
    # Try to match mode
    for mode in PerformanceMode:
        if mode.value == mode_name:
            return manager.get_preset(mode.value)
    
    # Default to balanced
    return manager.get_preset(PerformanceMode.BALANCED.value)
