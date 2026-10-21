"""
Measurement quality assessment module
Evaluates reliability of traffic counting based on capture FPS, inference speed, and data completeness
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional
from datetime import datetime


class MeasurementQuality(Enum):
    """Quality assessment levels"""
    EXCELLENT = "EXCELENTE"
    GOOD = "BUENO"
    LIMITED = "LIMITADO"
    UNRELIABLE = "NO CONFIABLE"


@dataclass
class QualityMetrics:
    """Quality assessment metrics"""
    camera_fps: float
    inference_fps: float
    capture_latency_ms: float
    inference_latency_ms: float
    total_pipeline_latency_ms: float
    
    frames_received: int
    frames_processed: int
    frames_dropped: int
    active_tracks: int
    reconnections: int
    
    # Thresholds (configurable)
    min_camera_fps: float = 10.0
    min_inference_fps: float = 5.0
    max_capture_latency_ms: float = 100.0
    max_inference_latency_ms: float = 200.0
    max_reconnections: int = 3
    
    def get_quality_level(self) -> MeasurementQuality:
        """Assess overall measurement quality"""
        issues = []
        
        # Check camera FPS
        if self.camera_fps < self.min_camera_fps:
            issues.append(f"Camera FPS too low: {self.camera_fps:.1f} (min: {self.min_camera_fps})")
        
        # Check inference FPS
        if self.inference_fps < self.min_inference_fps:
            issues.append(f"Inference FPS too low: {self.inference_fps:.1f} (min: {self.min_inference_fps})")
        
        # Check latencies
        if self.capture_latency_ms > self.max_capture_latency_ms:
            issues.append(f"Capture latency high: {self.capture_latency_ms:.1f}ms")
        
        if self.inference_latency_ms > self.max_inference_latency_ms:
            issues.append(f"Inference latency high: {self.inference_latency_ms:.1f}ms")
        
        # Check frame processing
        if self.frames_received > 0:
            drop_rate = self.frames_dropped / self.frames_received
            if drop_rate > 0.2:  # More than 20% dropped
                issues.append(f"Frame drop rate high: {drop_rate*100:.1f}%")
        
        # Check reconnections
        if self.reconnections > self.max_reconnections:
            issues.append(f"Too many reconnections: {self.reconnections}")
        
        # Determine quality level
        if len(issues) == 0:
            return MeasurementQuality.EXCELLENT
        elif len(issues) == 1:
            return MeasurementQuality.GOOD
        elif len(issues) <= 2:
            return MeasurementQuality.LIMITED
        else:
            return MeasurementQuality.UNRELIABLE
    
    def get_warnings(self) -> list:
        """Get list of quality warnings"""
        warnings = []
        
        if self.camera_fps < self.min_camera_fps:
            warnings.append({
                'type': 'LOW_CAMERA_FPS',
                'message': f'FPS DE CÁMARA BAJO ({self.camera_fps:.1f} FPS)',
                'description': 'RIESGO DE PÉRDIDA DE VEHÍCULOS RÁPIDOS',
                'severity': 'HIGH'
            })
        
        if self.inference_fps < self.min_inference_fps:
            warnings.append({
                'type': 'LOW_INFERENCE_FPS',
                'message': f'FPS DE IA BAJO ({self.inference_fps:.1f} FPS)',
                'description': 'Considere reducir resolución o usar modelo más rápido',
                'severity': 'MEDIUM'
            })
        
        if self.frames_received > 0 and self.frames_dropped / self.frames_received > 0.2:
            warnings.append({
                'type': 'HIGH_DROP_RATE',
                'message': f'TASA DE DESCARTE ALTA ({(self.frames_dropped/self.frames_received)*100:.1f}%)',
                'description': 'Frames se están descartando, pueden faltar conteos',
                'severity': 'HIGH'
            })
        
        if self.reconnections > 0:
            warnings.append({
                'type': 'RECONNECTIONS',
                'message': f'Reconexiones a cámara: {self.reconnections}',
                'description': 'Posible inestabilidad de conexión de red',
                'severity': 'MEDIUM'
            })
        
        if self.total_pipeline_latency_ms > 500:
            warnings.append({
                'type': 'HIGH_LATENCY',
                'message': f'Latencia alta: {self.total_pipeline_latency_ms:.0f}ms',
                'description': 'Sistema lento, considere optimizaciones',
                'severity': 'LOW'
            })
        
        return warnings
    
    def get_report(self) -> Dict:
        """Get comprehensive quality report"""
        return {
            'quality_level': self.get_quality_level().value,
            'warnings': self.get_warnings(),
            'metrics': {
                'camera_fps': f"{self.camera_fps:.1f} FPS",
                'inference_fps': f"{self.inference_fps:.1f} FPS",
                'capture_latency': f"{self.capture_latency_ms:.1f}ms",
                'inference_latency': f"{self.inference_latency_ms:.1f}ms",
                'total_latency': f"{self.total_pipeline_latency_ms:.1f}ms",
            },
            'data_integrity': {
                'frames_received': self.frames_received,
                'frames_processed': self.frames_processed,
                'frames_dropped': self.frames_dropped,
                'drop_rate': f"{(self.frames_dropped/max(self.frames_received, 1))*100:.1f}%",
                'active_tracks': self.active_tracks,
            },
            'connection': {
                'reconnections': self.reconnections,
            },
            'timestamp': datetime.now().isoformat(),
        }


class QualityMonitor:
    """Monitor measurement quality in real-time"""
    
    def __init__(self):
        self.metrics = None
        self.history = []  # Keep last 60 quality snapshots
    
    def update(self, camera_fps: float, inference_fps: float,
               capture_latency_ms: float, inference_latency_ms: float,
               total_pipeline_latency_ms: float, frames_received: int,
               frames_processed: int, frames_dropped: int,
               active_tracks: int, reconnections: int):
        """Update quality metrics"""
        self.metrics = QualityMetrics(
            camera_fps=camera_fps,
            inference_fps=inference_fps,
            capture_latency_ms=capture_latency_ms,
            inference_latency_ms=inference_latency_ms,
            total_pipeline_latency_ms=total_pipeline_latency_ms,
            frames_received=frames_received,
            frames_processed=frames_processed,
            frames_dropped=frames_dropped,
            active_tracks=active_tracks,
            reconnections=reconnections,
        )
        
        # Keep history
        self.history.append(self.metrics)
        if len(self.history) > 60:
            self.history.pop(0)
    
    def get_quality_level(self) -> Optional[MeasurementQuality]:
        """Get current quality level"""
        if self.metrics:
            return self.metrics.get_quality_level()
        return None
    
    def get_report(self) -> Optional[Dict]:
        """Get current quality report"""
        if self.metrics:
            return self.metrics.get_report()
        return None
    
    def get_average_metrics(self) -> Optional[Dict]:
        """Get average metrics from history"""
        if not self.history:
            return None
        
        avg_camera_fps = sum(m.camera_fps for m in self.history) / len(self.history)
        avg_inference_fps = sum(m.inference_fps for m in self.history) / len(self.history)
        avg_latency = sum(m.total_pipeline_latency_ms for m in self.history) / len(self.history)
        
        return {
            'avg_camera_fps': f"{avg_camera_fps:.1f} FPS",
            'avg_inference_fps': f"{avg_inference_fps:.1f} FPS",
            'avg_latency': f"{avg_latency:.1f}ms",
            'history_points': len(self.history),
        }
    
    def is_quality_acceptable(self) -> bool:
        """Quick check if quality is acceptable for counting"""
        if not self.metrics:
            return False
        
        quality = self.metrics.get_quality_level()
        return quality in [MeasurementQuality.EXCELLENT, MeasurementQuality.GOOD]


# Quick functions for GUI integration
def assess_quality(camera_fps: float, inference_fps: float, 
                   latency_ms: float, drop_rate: float) -> str:
    """Quick quality assessment"""
    if camera_fps < 10:
        return "⚠️ BAJO"
    elif camera_fps < 20 or inference_fps < 5:
        return "⚡ LIMITADO"
    else:
        return "✓ BUENO"


def get_quality_color(quality_level: MeasurementQuality) -> str:
    """Get color code for quality level"""
    colors = {
        MeasurementQuality.EXCELLENT: "#00AA00",  # Green
        MeasurementQuality.GOOD: "#AAAA00",       # Yellow-green
        MeasurementQuality.LIMITED: "#FFAA00",     # Orange
        MeasurementQuality.UNRELIABLE: "#AA0000",  # Red
    }
    return colors.get(quality_level, "#AAAAAA")
