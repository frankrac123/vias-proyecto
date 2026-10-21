"""
Performance profiling and diagnostics module
Tracks latencies and performance metrics across the pipeline
"""
import time
from typing import Dict, Optional
from collections import deque
import threading


class PerformanceMetrics:
    """Track performance metrics with time windows."""
    
    def __init__(self, window_size: int = 60):
        """
        Initialize performance metrics tracker.
        
        Args:
            window_size: Number of samples to keep for averaging
        """
        self.window_size = window_size
        self.lock = threading.Lock()
        
        # Metric deques (keep last N samples)
        self.capture_times = deque(maxlen=window_size)
        self.inference_times = deque(maxlen=window_size)
        self.tracking_times = deque(maxlen=window_size)
        self.analytics_times = deque(maxlen=window_size)
        self.render_times = deque(maxlen=window_size)
        
        # Overall statistics
        self.total_frames_processed = 0
        self.total_frames_dropped = 0
        self.start_time = time.time()
    
    def record_capture(self, duration_ms: float):
        """Record frame capture duration."""
        with self.lock:
            self.capture_times.append(duration_ms)
    
    def record_inference(self, duration_ms: float):
        """Record YOLO inference duration."""
        with self.lock:
            self.inference_times.append(duration_ms)
    
    def record_tracking(self, duration_ms: float):
        """Record tracking update duration."""
        with self.lock:
            self.tracking_times.append(duration_ms)
    
    def record_analytics(self, duration_ms: float):
        """Record analytics processing duration."""
        with self.lock:
            self.analytics_times.append(duration_ms)
    
    def record_render(self, duration_ms: float):
        """Record rendering/display duration."""
        with self.lock:
            self.render_times.append(duration_ms)
    
    def record_frame_processed(self):
        """Increment processed frame counter."""
        with self.lock:
            self.total_frames_processed += 1
    
    def record_frame_dropped(self):
        """Increment dropped frame counter."""
        with self.lock:
            self.total_frames_dropped += 1
    
    def get_average(self, metric_deque: deque) -> float:
        """Calculate average of metric values."""
        if len(metric_deque) == 0:
            return 0.0
        return sum(metric_deque) / len(metric_deque)
    
    def get_max(self, metric_deque: deque) -> float:
        """Get maximum metric value."""
        if len(metric_deque) == 0:
            return 0.0
        return max(metric_deque)
    
    def get_min(self, metric_deque: deque) -> float:
        """Get minimum metric value."""
        if len(metric_deque) == 0:
            return 0.0
        return min(metric_deque)
    
    def get_percentile(self, metric_deque: deque, percentile: float) -> float:
        """Get percentile value of metric."""
        if len(metric_deque) == 0:
            return 0.0
        sorted_metrics = sorted(list(metric_deque))
        idx = int(len(sorted_metrics) * percentile / 100)
        return sorted_metrics[min(idx, len(sorted_metrics) - 1)]
    
    def get_report(self) -> Dict:
        """Get comprehensive performance report."""
        with self.lock:
            elapsed_time = time.time() - self.start_time
            
            return {
                'elapsed_seconds': elapsed_time,
                'total_frames_processed': self.total_frames_processed,
                'total_frames_dropped': self.total_frames_dropped,
                'overall_fps': self.total_frames_processed / elapsed_time if elapsed_time > 0 else 0,
                
                'capture': {
                    'avg_ms': self.get_average(self.capture_times),
                    'max_ms': self.get_max(self.capture_times),
                    'min_ms': self.get_min(self.capture_times),
                    'p95_ms': self.get_percentile(self.capture_times, 95),
                },
                
                'inference': {
                    'avg_ms': self.get_average(self.inference_times),
                    'max_ms': self.get_max(self.inference_times),
                    'min_ms': self.get_min(self.inference_times),
                    'p95_ms': self.get_percentile(self.inference_times, 95),
                    'avg_fps': 1000.0 / max(self.get_average(self.inference_times), 1) if len(self.inference_times) > 0 else 0,
                },
                
                'tracking': {
                    'avg_ms': self.get_average(self.tracking_times),
                    'max_ms': self.get_max(self.tracking_times),
                    'min_ms': self.get_min(self.tracking_times),
                    'p95_ms': self.get_percentile(self.tracking_times, 95),
                },
                
                'analytics': {
                    'avg_ms': self.get_average(self.analytics_times),
                    'max_ms': self.get_max(self.analytics_times),
                    'min_ms': self.get_min(self.analytics_times),
                    'p95_ms': self.get_percentile(self.analytics_times, 95),
                },
                
                'render': {
                    'avg_ms': self.get_average(self.render_times),
                    'max_ms': self.get_max(self.render_times),
                    'min_ms': self.get_min(self.render_times),
                    'p95_ms': self.get_percentile(self.render_times, 95),
                },
                
                'pipeline_total': {
                    'avg_ms': (self.get_average(self.capture_times) + 
                               self.get_average(self.inference_times) + 
                               self.get_average(self.tracking_times) + 
                               self.get_average(self.analytics_times)),
                }
            }
    
    def print_report(self):
        """Print formatted performance report."""
        report = self.get_report()
        
        print("\n" + "="*70)
        print("PERFORMANCE REPORT")
        print("="*70)
        print(f"Elapsed Time: {report['elapsed_seconds']:.1f}s")
        print(f"Frames Processed: {report['total_frames_processed']}")
        print(f"Frames Dropped: {report['total_frames_dropped']}")
        print(f"Overall FPS: {report['overall_fps']:.1f}")
        print()
        
        print("CAPTURE (Frame Acquisition)")
        cap = report['capture']
        print(f"  Avg: {cap['avg_ms']:.2f}ms | Max: {cap['max_ms']:.2f}ms | P95: {cap['p95_ms']:.2f}ms")
        
        print("INFERENCE (YOLO Detection)")
        inf = report['inference']
        print(f"  Avg: {inf['avg_ms']:.2f}ms | Max: {inf['max_ms']:.2f}ms | P95: {inf['p95_ms']:.2f}ms")
        print(f"  Avg FPS: {inf['avg_fps']:.1f}")
        
        print("TRACKING (ByteTrack Update)")
        track = report['tracking']
        print(f"  Avg: {track['avg_ms']:.2f}ms | Max: {track['max_ms']:.2f}ms | P95: {track['p95_ms']:.2f}ms")
        
        print("ANALYTICS (Line Crossing Detection)")
        ana = report['analytics']
        print(f"  Avg: {ana['avg_ms']:.2f}ms | Max: {ana['max_ms']:.2f}ms | P95: {ana['p95_ms']:.2f}ms")
        
        print("RENDER (GUI Display)")
        ren = report['render']
        print(f"  Avg: {ren['avg_ms']:.2f}ms | Max: {ren['max_ms']:.2f}ms | P95: {ren['p95_ms']:.2f}ms")
        
        print("\nPIPELINE TOTAL (excl. render)")
        pipe = report['pipeline_total']
        print(f"  Avg: {pipe['avg_ms']:.2f}ms")
        print(f"  Potential FPS: {1000.0/max(pipe['avg_ms'], 1):.1f}")
        
        print("="*70 + "\n")


class TimerContext:
    """Context manager for timing code blocks."""
    
    def __init__(self, metrics: PerformanceMetrics, metric_type: str):
        """
        Initialize timer.
        
        Args:
            metrics: PerformanceMetrics instance
            metric_type: Type of metric ('capture', 'inference', etc)
        """
        self.metrics = metrics
        self.metric_type = metric_type
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed_ms = (time.time() - self.start_time) * 1000.0
        
        if self.metric_type == 'capture':
            self.metrics.record_capture(elapsed_ms)
        elif self.metric_type == 'inference':
            self.metrics.record_inference(elapsed_ms)
        elif self.metric_type == 'tracking':
            self.metrics.record_tracking(elapsed_ms)
        elif self.metric_type == 'analytics':
            self.metrics.record_analytics(elapsed_ms)
        elif self.metric_type == 'render':
            self.metrics.record_render(elapsed_ms)


# Global metrics instance
_metrics = PerformanceMetrics()


def get_metrics() -> PerformanceMetrics:
    """Get global metrics instance."""
    return _metrics


def timer(metric_type: str):
    """Decorator for timing functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with TimerContext(_metrics, metric_type):
                return func(*args, **kwargs)
        return wrapper
    return decorator
