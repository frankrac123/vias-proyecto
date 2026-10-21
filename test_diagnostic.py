#!/usr/bin/env python3
"""
Diagnostic script for testing the traffic detection pipeline
Tests video capture, frame processing, and latency measurements
"""
import time
import argparse
from app.video_capture import AsyncVideoCapture
from app.detector import TrafficDetector
from app.tracker import TrackManager
from app.counting import LineIntersectionDetector, CountingLine
from app.profiling import PerformanceMetrics, TimerContext
import numpy as np


def test_video_capture(source: str, source_type: str = "auto", duration: int = 10):
    """Test video capture."""
    print(f"\n{'='*70}")
    print("TEST 1: VIDEO CAPTURE")
    print(f"{'='*70}")
    print(f"Source: {source}")
    print(f"Type: {source_type}")
    print(f"Duration: {duration}s\n")
    
    cap = AsyncVideoCapture(source, source_type=source_type)
    
    if not cap.start():
        print("❌ Failed to start capture")
        return False
    
    print("✓ Capture started")
    time.sleep(1)  # Wait for connection
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if ret and frame is not None:
                frame_count += 1
                if frame_count % 10 == 0:
                    status = cap.get_status()
                    print(f"  Frames: {frame_count} | FPS: {status['fps']:.1f} | Connected: {status['connected']}")
            time.sleep(0.01)
    finally:
        cap.stop()
    
    status = cap.get_status()
    total_time = time.time() - start_time
    actual_fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"\n✓ Capture Test Complete")
    print(f"  Total Frames: {frame_count}")
    print(f"  Duration: {total_time:.1f}s")
    print(f"  Average FPS: {actual_fps:.1f}")
    print(f"  Reported FPS: {status['fps']:.1f}")
    print(f"  Resolution: {status['width']}x{status['height']}")
    
    if actual_fps < 5:
        print(f"  ⚠️  WARNING: FPS is very low ({actual_fps:.1f})")
    
    return actual_fps > 0


def test_inference(model_path: str = "yolo11n.pt", num_frames: int = 5):
    """Test YOLO inference speed."""
    print(f"\n{'='*70}")
    print("TEST 2: YOLO INFERENCE")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Test Frames: {num_frames}\n")
    
    detector = TrafficDetector(model_path=model_path, confidence=0.3, imgsz=416)
    
    if not detector.is_loaded:
        print("❌ Failed to load model")
        return False
    
    print("✓ Model loaded")
    
    # Create a dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    inference_times = []
    
    for i in range(num_frames):
        start = time.time()
        result = detector.detect(dummy_frame)
        elapsed = time.time() - start
        
        if result.get('success'):
            inference_times.append(result.get('inference_time', 0) * 1000)  # Convert to ms
            print(f"  Frame {i+1}: {inference_times[-1]:.2f}ms")
        else:
            print(f"  Frame {i+1}: Failed")
    
    if inference_times:
        avg_time = np.mean(inference_times)
        avg_fps = 1000 / avg_time if avg_time > 0 else 0
        
        print(f"\n✓ Inference Test Complete")
        print(f"  Average Time: {avg_time:.2f}ms")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Min/Max: {min(inference_times):.2f}ms / {max(inference_times):.2f}ms")
        return True
    
    return False


def test_tracking(num_frames: int = 10):
    """Test tracking speed."""
    print(f"\n{'='*70}")
    print("TEST 3: TRACKING")
    print(f"{'='*70}")
    print(f"Test Frames: {num_frames}\n")
    
    tracker = TrackManager()
    metrics = PerformanceMetrics()
    
    tracking_times = []
    
    # Simulate detections
    for frame_id in range(num_frames):
        # Create dummy detections
        detections = [
            {
                'id': 0,
                'class_id': 2,
                'class_name': 'car',
                'class_label': 'Carro',
                'confidence': 0.9,
                'bbox': [100 + frame_id*10, 200, 250 + frame_id*10, 350],
                'center_x': 175 + frame_id*10,
                'center_y': 275
            },
            {
                'id': 1,
                'class_id': 0,
                'class_name': 'person',
                'class_label': 'Persona',
                'confidence': 0.85,
                'bbox': [400, 100, 450, 300],
                'center_x': 425,
                'center_y': 200
            }
        ]
        
        with TimerContext(metrics, 'tracking'):
            tracker.update(detections, frame_id)
        
        tracking_times.append(metrics.tracking_times[-1] if metrics.tracking_times else 0)
        
        if (frame_id + 1) % 5 == 0:
            print(f"  Frame {frame_id + 1}: {tracking_times[-1]:.2f}ms")
    
    report = metrics.get_report()
    print(f"\n✓ Tracking Test Complete")
    print(f"  Average Time: {report['tracking']['avg_ms']:.2f}ms")
    print(f"  Max Time: {report['tracking']['max_ms']:.2f}ms")
    
    return True


def test_full_pipeline(source: str, duration: int = 15):
    """Test full pipeline with actual video capture."""
    print(f"\n{'='*70}")
    print("TEST 4: FULL PIPELINE")
    print(f"{'='*70}")
    print(f"Source: {source}")
    print(f"Duration: {duration}s\n")
    
    cap = AsyncVideoCapture(source)
    detector = TrafficDetector()
    tracker = TrackManager()
    
    # Create a dummy counting line
    line = CountingLine(
        name="Test Line",
        norm_x1=0.0, norm_y1=0.5,
        norm_x2=1.0, norm_y2=0.5,
        frame_height=480, frame_width=640
    )
    line_detector = LineIntersectionDetector([line])
    
    metrics = PerformanceMetrics()
    
    if not cap.start():
        print("❌ Failed to start capture")
        return False
    
    print("✓ Pipeline initialized")
    time.sleep(1)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            
            if ret and frame is not None:
                # Inference
                with TimerContext(metrics, 'inference'):
                    result = detector.detect(frame)
                
                if result.get('success'):
                    detections = result.get('detections', [])
                    
                    # Tracking
                    with TimerContext(metrics, 'tracking'):
                        tracker.update(detections, frame_count)
                    
                    # Analytics
                    with TimerContext(metrics, 'analytics'):
                        tracks = tracker.get_tracks()
                        events = line_detector.process_tracks(tracks, frame_count)
                    
                    metrics.record_frame_processed()
                    frame_count += 1
                    
                    if frame_count % 20 == 0:
                        report = metrics.get_report()
                        print(f"  Frames: {frame_count} | Inference: {report['inference']['avg_ms']:.1f}ms | "
                              f"Total Pipeline: {report['pipeline_total']['avg_ms']:.1f}ms")
            
            time.sleep(0.001)
    
    finally:
        cap.stop()
    
    report = metrics.get_report()
    
    print(f"\n✓ Full Pipeline Test Complete")
    print(f"  Total Frames: {frame_count}")
    print(f"  Capture FPS: {cap.get_status()['fps']:.1f}")
    print(f"  Average Inference: {report['inference']['avg_ms']:.2f}ms ({report['inference']['avg_fps']:.1f} FPS)")
    print(f"  Average Tracking: {report['tracking']['avg_ms']:.2f}ms")
    print(f"  Average Analytics: {report['analytics']['avg_ms']:.2f}ms")
    print(f"  Total Pipeline: {report['pipeline_total']['avg_ms']:.2f}ms ({1000/max(report['pipeline_total']['avg_ms'], 1):.1f} FPS potential)")
    
    return frame_count > 0


def main():
    parser = argparse.ArgumentParser(description='Traffic Detection Diagnostic Tests')
    parser.add_argument('--test', type=str, default='all', 
                       choices=['capture', 'inference', 'tracking', 'full', 'all'],
                       help='Test to run')
    parser.add_argument('--source', type=str, default=None,
                       help='Video source (defaults to config/config.json)')
    parser.add_argument('--type', type=str, default='auto',
                       choices=['auto', 'mjpeg', 'rtsp', 'http', 'file', 'usb'],
                       help='Source type')
    parser.add_argument('--duration', type=int, default=10,
                       help='Test duration in seconds')
    
    args = parser.parse_args()

    if args.source is None:
        from app.config import ConfigManager
        camera_config = ConfigManager("./config").get_config().camera
        args.source = camera_config.source
        if args.type == "auto":
            args.type = camera_config.source_type
    
    print(f"\n{'='*70}")
    print("TRAFFIC DETECTION SYSTEM DIAGNOSTIC TESTS")
    print(f"{'='*70}")
    
    results = {}
    
    try:
        if args.test in ['capture', 'all']:
            results['capture'] = test_video_capture(args.source, args.type, args.duration)
        
        if args.test in ['inference', 'all']:
            results['inference'] = test_inference()
        
        if args.test in ['tracking', 'all']:
            results['tracking'] = test_tracking()
        
        if args.test in ['full', 'all']:
            results['full'] = test_full_pipeline(args.source, args.duration)
        
        # Summary
        print(f"\n{'='*70}")
        print("TEST SUMMARY")
        print(f"{'='*70}")
        for test_name, success in results.items():
            status = "✓ PASS" if success else "❌ FAIL"
            print(f"{test_name.upper()}: {status}")
        
        all_passed = all(results.values())
        if all_passed:
            print("\n✓ ALL TESTS PASSED")
        else:
            print("\n⚠️  SOME TESTS FAILED - See details above")
        return all_passed

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
