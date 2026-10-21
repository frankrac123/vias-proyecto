#!/usr/bin/env python3
"""Quick profiling test"""
from app.profiling import PerformanceMetrics, TimerContext
import time

m = PerformanceMetrics()

for i in range(5):
    with TimerContext(m, 'inference'):
        time.sleep(0.05)
    m.record_frame_processed()

report = m.get_report()
print('✓ Profiling test:')
print(f'  Inference avg: {report["inference"]["avg_ms"]:.2f}ms')
print(f'  Frames processed: {report["total_frames_processed"]}')
