"""
PySide6-based GUI for Traffic Detection System
Main application window with video display, analytics, and controls
"""
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSpinBox, QDoubleSpinBox, QComboBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QDialog, QCheckBox, QLineEdit, QFormLayout, QMessageBox,
    QFileDialog, QProgressBar, QGroupBox, QGridLayout, QSplitter
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QSize, Slot
from PySide6.QtGui import QImage, QPixmap, QIcon, QFont, QColor
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QDateTimeAxis, QValueAxis
from PySide6.QtCore import QDateTime, QPoint

import cv2
import numpy as np
import threading
import time
from typing import Optional, Dict, List
from datetime import datetime, timedelta

from app.video_capture import AsyncVideoCapture
from app.detector import TrafficDetector
from app.tracker import TrackManager, Track
from app.counting import LineIntersectionDetector, CountingLine
from app.database import DatabaseManager
from app.analytics import AnalyticsEngine
from app.exporter import DataExporter
from app.config import ConfigManager
from app.profiling import PerformanceMetrics, TimerContext
from app.quality_monitor import QualityMonitor, get_quality_color
from app.crossing_engine import CrossingEngine, CrossingLine as V3CrossingLine
import app.utils as utils


class ProcessingThread(QThread):
    """Separate thread for video processing"""
    
    frame_ready = Signal(object)  # Signal emits dict with results
    error_occurred = Signal(str)
    
    def __init__(self, video_capture: AsyncVideoCapture,
                 detector: TrafficDetector,
                 track_manager: TrackManager,
                 line_detector: LineIntersectionDetector,
                 analytics_engine: AnalyticsEngine,
                 database: DatabaseManager,
                 session_id: int,
                 crossing_engine: CrossingEngine):
        super().__init__()
        
        self.video_capture = video_capture
        self.detector = detector
        self.track_manager = track_manager
        self.line_detector = line_detector
        self.analytics_engine = analytics_engine
        self.database = database
        self.session_id = session_id
        self.crossing_engine = crossing_engine
        
        self.is_running = False
        self.frame_count = 0
        self.total_frames_processed = 0
        self.fps = 0.0
        self.last_fps_update = time.time()
        self.inference_times = []
        
        # Performance metrics
        self.metrics = PerformanceMetrics(window_size=60)
        self.quality_monitor = QualityMonitor()
    
    def run(self):
        """Main processing loop"""
        self.is_running = True
        last_processed_frame_count = -1
        
        while self.is_running:
            try:
                # Get latest frame (should already be in buffer)
                ret, frame = self.video_capture.read()
                
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                
                # FRAMESKIP: Only process if we have a new frame since last processing
                # This prevents processing stale frames when YOLO is slow
                current_frame_count = self.video_capture.total_frames_captured
                if current_frame_count == last_processed_frame_count:
                    time.sleep(0.001)
                    continue
                
                last_processed_frame_count = current_frame_count
                
                # Measure inference time
                with TimerContext(self.metrics, 'inference'):
                    detection_result = self.detector.detect(frame)
                
                if not detection_result.get('success', False):
                    time.sleep(0.01)
                    continue
                
                detections = detection_result.get('detections', [])
                self.frame_count += 1
                self.total_frames_processed += 1
                self.metrics.record_frame_processed()
                
                inference_time = detection_result.get('inference_time', 0)
                self.inference_times.append(inference_time)
                
                # Keep only recent inference times
                if len(self.inference_times) > 30:
                    self.inference_times.pop(0)
                
                # Measure tracking time
                with TimerContext(self.metrics, 'tracking'):
                    self.track_manager.update(detections, self.total_frames_processed)
                
                # Measure analytics time
                with TimerContext(self.metrics, 'analytics'):
                    tracks = self.track_manager.get_tracks()
                    h, w = frame.shape[:2]
                    self.crossing_engine.set_lines([
                        V3CrossingLine(
                            line_id=line.line_id, name=line.name,
                            p1=(line.norm_x1 * w, line.norm_y1 * h),
                            p2=(line.norm_x2 * w, line.norm_y2 * h),
                            side_a_name=line.direction_a,
                            side_b_name=line.direction_b
                        )
                        for line in self.line_detector.get_lines().values() if line.enabled
                    ])
                    crossing_events = self.crossing_engine.process_tracks(
                        tracks, self.total_frames_processed,
                        camera_fps=self.video_capture.get_fps()
                    )
                    
                    # Add events to database and analytics
                    for event in crossing_events:
                        event_dict = event.to_dict()
                        event_dict.update({
                            'track_id': event.tracker_id, 'class_label': event.class_name,
                            'direction': event.direction, 'position_x': event.x,
                            'position_y': event.y
                        })
                        
                        self.database.add_crossing_event(self.session_id, event_dict)
                        self.analytics_engine.add_event(event_dict)
                
                # Calculate FPS
                current_time = time.time()
                if current_time - self.last_fps_update >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_fps_update)
                    self.frame_count = 0
                    self.last_fps_update = current_time
                
                # Emit results
                capture_status = self.video_capture.get_status()
                inference_fps = 1.0 / max(
                    np.mean(self.inference_times) if self.inference_times else 0.0,
                    0.001
                )
                frames_received = capture_status.get('total_frames_captured', 0)
                frames_dropped = max(frames_received - self.total_frames_processed, 0)
                inference_ms = (
                    np.mean(self.inference_times) * 1000
                    if self.inference_times else 0.0
                )
                self.quality_monitor.update(
                    camera_fps=self.video_capture.get_fps(),
                    inference_fps=inference_fps,
                    capture_latency_ms=0.0,
                    inference_latency_ms=inference_ms,
                    total_pipeline_latency_ms=inference_ms,
                    frames_received=frames_received,
                    frames_processed=self.total_frames_processed,
                    frames_dropped=frames_dropped,
                    active_tracks=len(tracks),
                    reconnections=max(capture_status.get('connection_attempts', 1) - 1, 0),
                )
                result = {
                    'frame': frame,
                    'detections': detections,
                    'tracks': tracks,
                    'crossing_events': crossing_events,
                    'fps': self.fps,
                    'inference_time': np.mean(self.inference_times) if self.inference_times else 0,
                    'camera_fps': self.video_capture.get_fps(),
                    'metrics': self.metrics.get_report()
                    ,'quality': self.quality_monitor.get_report()
                }
                
                self.frame_ready.emit(result)
                
            except Exception as e:
                self.error_occurred.emit(str(e))
                time.sleep(0.1)
    
    def stop(self):
        """Stop processing thread and print performance report"""
        self.is_running = False
        self.wait()
        # Print final performance report
        print("\n" + "="*70)
        print("PROCESSING THREAD PERFORMANCE REPORT")
        print("="*70)
        report = self.metrics.get_report()
        print(f"Total Frames Processed: {report['total_frames_processed']}")
        print(f"Total Elapsed: {report['elapsed_seconds']:.1f}s")
        print(f"Average Inference: {report['inference']['avg_ms']:.2f}ms ({report['inference']['avg_fps']:.1f} FPS)")
        print(f"Average Total Pipeline: {report['pipeline_total']['avg_ms']:.2f}ms ({1000.0/max(report['pipeline_total']['avg_ms'], 1):.1f} FPS potential)")
        print("="*70)


class LineEditorDialog(QDialog):
    """Dialog for editing counting lines"""
    
    def __init__(self, parent=None, line: Optional[CountingLine] = None, 
                 frame_shape: tuple = None):
        super().__init__(parent)
        
        self.line = line
        self.frame_shape = frame_shape or (480, 640)
        self.setWindowTitle("Editor de Línea de Conteo")
        self.setGeometry(100, 100, 600, 500)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QFormLayout()
        
        # Name
        self.name_input = QLineEdit()
        if self.line:
            self.name_input.setText(self.line.name)
        else:
            self.name_input.setText("Línea de Conteo")
        layout.addRow("Nombre:", self.name_input)

        self.direction_a_input = QLineEdit(self.line.direction_a if self.line else "A")
        self.direction_b_input = QLineEdit(self.line.direction_b if self.line else "B")
        layout.addRow("Nombre lado A:", self.direction_a_input)
        layout.addRow("Nombre lado B:", self.direction_b_input)
        
        # Normalized coordinates
        layout.addRow(QLabel("Coordenadas Normalizadas (0.0 - 1.0):"))
        
        self.x1_spinbox = QDoubleSpinBox()
        self.x1_spinbox.setRange(0, 1.0)
        self.x1_spinbox.setSingleStep(0.01)
        self.x1_spinbox.setValue(self.line.norm_x1 if self.line else 0.5)
        layout.addRow("X1:", self.x1_spinbox)
        
        self.y1_spinbox = QDoubleSpinBox()
        self.y1_spinbox.setRange(0, 1.0)
        self.y1_spinbox.setSingleStep(0.01)
        self.y1_spinbox.setValue(self.line.norm_y1 if self.line else 0.0)
        layout.addRow("Y1:", self.y1_spinbox)
        
        self.x2_spinbox = QDoubleSpinBox()
        self.x2_spinbox.setRange(0, 1.0)
        self.x2_spinbox.setSingleStep(0.01)
        self.x2_spinbox.setValue(self.line.norm_x2 if self.line else 0.5)
        layout.addRow("X2:", self.x2_spinbox)
        
        self.y2_spinbox = QDoubleSpinBox()
        self.y2_spinbox.setRange(0, 1.0)
        self.y2_spinbox.setSingleStep(0.01)
        self.y2_spinbox.setValue(self.line.norm_y2 if self.line else 1.0)
        layout.addRow("Y2:", self.y2_spinbox)
        
        # Preset buttons
        buttons_layout = QHBoxLayout()
        
        btn_horizontal = QPushButton("Línea Horizontal")
        btn_horizontal.clicked.connect(self.set_horizontal)
        buttons_layout.addWidget(btn_horizontal)
        
        btn_vertical = QPushButton("Línea Vertical")
        btn_vertical.clicked.connect(self.set_vertical)
        buttons_layout.addWidget(btn_vertical)
        
        btn_diagonal = QPushButton("Línea Diagonal")
        btn_diagonal.clicked.connect(self.set_diagonal)
        buttons_layout.addWidget(btn_diagonal)
        
        layout.addRow(buttons_layout)
        
        # OK/Cancel buttons
        button_layout = QHBoxLayout()
        
        btn_ok = QPushButton("Aceptar")
        btn_ok.clicked.connect(self.accept)
        button_layout.addWidget(btn_ok)
        
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(btn_cancel)
        
        layout.addRow(button_layout)
        
        self.setLayout(layout)
    
    def set_horizontal(self):
        """Set horizontal line (middle of frame)"""
        self.y1_spinbox.setValue(0.5)
        self.y2_spinbox.setValue(0.5)
        self.x1_spinbox.setValue(0.0)
        self.x2_spinbox.setValue(1.0)
    
    def set_vertical(self):
        """Set vertical line (middle of frame)"""
        self.x1_spinbox.setValue(0.5)
        self.x2_spinbox.setValue(0.5)
        self.y1_spinbox.setValue(0.0)
        self.y2_spinbox.setValue(1.0)
    
    def set_diagonal(self):
        """Set diagonal line"""
        self.x1_spinbox.setValue(0.0)
        self.y1_spinbox.setValue(0.0)
        self.x2_spinbox.setValue(1.0)
        self.y2_spinbox.setValue(1.0)
    
    def get_line_data(self):
        """Get line data from dialog"""
        return {
            'name': self.name_input.text(),
            'direction_a': self.direction_a_input.text().strip() or 'A',
            'direction_b': self.direction_b_input.text().strip() or 'B',
            'norm_x1': self.x1_spinbox.value(),
            'norm_y1': self.y1_spinbox.value(),
            'norm_x2': self.x2_spinbox.value(),
            'norm_y2': self.y2_spinbox.value()
        }


class SettingsDialog(QDialog):
    """Application settings dialog"""
    
    def __init__(self, parent=None, config_manager: ConfigManager = None):
        super().__init__(parent)
        
        self.config_manager = config_manager
        self.config = config_manager.config if config_manager else None
        
        self.setWindowTitle("Configuración")
        self.setGeometry(100, 100, 600, 700)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        
        # Camera settings
        camera_group = QGroupBox("Cámara")
        camera_layout = QFormLayout()
        
        self.camera_url = QLineEdit()
        if self.config:
            self.camera_url.setText(self.config.camera.source)
        camera_layout.addRow("URL Cámara:", self.camera_url)
        
        self.camera_type = QComboBox()
        self.camera_type.addItems(["mjpeg", "rtsp", "file", "usb"])
        if self.config:
            self.camera_type.setCurrentText(self.config.camera.source_type)
        camera_layout.addRow("Tipo de Fuente:", self.camera_type)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # Detection settings
        detection_group = QGroupBox("Detección")
        detection_layout = QFormLayout()
        
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.0, 1.0)
        self.confidence.setSingleStep(0.05)
        if self.config:
            self.confidence.setValue(self.config.detection.confidence)
        detection_layout.addRow("Confianza:", self.confidence)
        
        self.imgsz = QSpinBox()
        self.imgsz.setRange(320, 1280)
        if self.config:
            self.imgsz.setValue(self.config.detection.imgsz)
        detection_layout.addRow("Tamaño de imagen:", self.imgsz)
        
        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "0", "1"])
        if self.config:
            self.device.setCurrentText(self.config.detection.device)
        detection_layout.addRow("Dispositivo:", self.device)
        
        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)
        
        # Performance mode
        performance_group = QGroupBox("Modo de Rendimiento")
        performance_layout = QFormLayout()
        
        self.performance_mode = QComboBox()
        self.performance_mode.addItems(["quality", "balanced", "performance"])
        if self.config:
            self.performance_mode.setCurrentText(self.config.performance_mode)
        performance_layout.addRow("Modo:", self.performance_mode)
        
        performance_group.setLayout(performance_layout)
        layout.addWidget(performance_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        btn_ok = QPushButton("Guardar")
        btn_ok.clicked.connect(self.accept)
        button_layout.addWidget(btn_ok)
        
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(btn_cancel)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def get_settings(self):
        """Get settings from dialog"""
        return {
            'camera_url': self.camera_url.text(),
            'camera_type': self.camera_type.currentText(),
            'confidence': self.confidence.value(),
            'imgsz': self.imgsz.value(),
            'device': self.device.currentText(),
            'performance_mode': self.performance_mode.currentText()
        }


class TrafficDetectionApp(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Sistema de Aforo Vehicular - Traffic Detection")
        self.setGeometry(0, 0, 1400, 900)
        
        # Initialize managers
        self.config_manager = ConfigManager("./config")
        self.config = self.config_manager.get_config()
        
        self.database = DatabaseManager()
        self.session_id = self.database.create_session(
            camera_source=self.config.camera.source,
            notes="Auto-created session"
        )
        
        self.detector = TrafficDetector(
            model_path=self.config.detection.model_path,
            confidence=self.config.detection.confidence,
            imgsz=self.config.detection.imgsz,
            device=self.config.detection.device
        )
        
        self.track_manager = TrackManager(max_trajectory_length=self.config.analytics.trajectory_max_length)
        self.line_detector = LineIntersectionDetector()
        self.crossing_engine = CrossingEngine(session_id=self.session_id)
        self.analytics_engine = AnalyticsEngine(intervals=self.config.analytics.intervals)
        
        self.video_capture = AsyncVideoCapture(
            source=self.config.camera.source,
            source_type=self.config.camera.source_type
        )
        
        self.exporter = DataExporter(export_path=self.config.export_path)
        
        # Add default counting line
        self.line_detector.add_line(
            self.config.line.x1, self.config.line.y1,
            self.config.line.x2, self.config.line.y2,
            name=self.config.line.name,
            norm_coords=True
        )
        
        # Processing thread
        self.processing_thread = None
        
        # UI state
        self.is_running = False
        self.current_frame = None
        self.crossing_badges = []
        self.last_frame_time = time.time()
        
        # Initialize UI
        self.init_ui()
        
        # Timer for display updates
        self.display_timer = QTimer()
        self.display_timer.timeout.connect(self.update_display)
    
    def init_ui(self):
        """Initialize user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel: Video display
        left_layout = QVBoxLayout()
        
        # Video label
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid gray;")
        self.video_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(QLabel("VISTA EN VIVO"), 0, Qt.AlignTop)
        left_layout.addWidget(self.video_label, 1)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        self.btn_start = QPushButton("▶ INICIAR")
        self.btn_start.setStyleSheet("background-color: green; color: white; font-weight: bold; padding: 10px;")
        self.btn_start.clicked.connect(self.start_capture)
        buttons_layout.addWidget(self.btn_start)
        
        self.btn_stop = QPushButton("⏹ DETENER")
        self.btn_stop.setStyleSheet("background-color: red; color: white; font-weight: bold; padding: 10px;")
        self.btn_stop.clicked.connect(self.stop_capture)
        self.btn_stop.setEnabled(False)
        buttons_layout.addWidget(self.btn_stop)
        
        self.btn_settings = QPushButton("⚙ CONFIGURACIÓN")
        self.btn_settings.clicked.connect(self.open_settings)
        buttons_layout.addWidget(self.btn_settings)
        
        self.btn_line_editor = QPushButton("📏 EDITAR LÍNEA")
        self.btn_line_editor.clicked.connect(self.open_line_editor)
        buttons_layout.addWidget(self.btn_line_editor)
        
        left_layout.addLayout(buttons_layout)
        
        main_layout.addLayout(left_layout, 2)
        
        # Right panel: Stats and controls
        right_layout = QVBoxLayout()
        
        # Tabs
        self.tabs = QTabWidget()
        
        # Tab 1: Real-time stats
        self.tab_stats = QWidget()
        self.init_stats_tab()
        self.tabs.addTab(self.tab_stats, "📊 Estadísticas")
        
        # Tab 2: Detailed events
        self.tab_events = QWidget()
        self.init_events_tab()
        self.tabs.addTab(self.tab_events, "📋 Eventos")
        
        # Tab 3: Export
        self.tab_export = QWidget()
        self.init_export_tab()
        self.tabs.addTab(self.tab_export, "📤 Exportar")
        
        right_layout.addWidget(self.tabs)
        
        main_layout.addLayout(right_layout, 1)
    
    def init_stats_tab(self):
        """Initialize statistics tab"""
        layout = QVBoxLayout(self.tab_stats)
        
        # Real-time counters
        layout.addWidget(QLabel("ACTIVOS AHORA"))
        self.label_active_objects = QLabel("Persona: 0   Carro: 0   Moto: 0")
        self.label_active_objects.setStyleSheet(
            "font-size: 15px; color: #67d5ff; background-color: #18222e; padding: 9px;"
        )
        layout.addWidget(self.label_active_objects)
        layout.addWidget(QLabel("CRUCES REGISTRADOS"))
        counters_layout = QGridLayout()
        
        # Total
        self.label_total = QLabel("0")
        self.label_total.setStyleSheet("font-size: 24px; font-weight: bold; color: white; background-color: #333; padding: 10px; border-radius: 5px;")
        counters_layout.addWidget(QLabel("TOTAL:"), 0, 0)
        counters_layout.addWidget(self.label_total, 0, 1)
        
        # Vehicles
        self.label_vehicles = QLabel("0")
        self.label_vehicles.setStyleSheet("font-size: 24px; font-weight: bold; color: #00FF00; background-color: #333; padding: 10px; border-radius: 5px;")
        counters_layout.addWidget(QLabel("VEHÍCULOS:"), 1, 0)
        counters_layout.addWidget(self.label_vehicles, 1, 1)
        
        # Pedestrians
        self.label_pedestrians = QLabel("0")
        self.label_pedestrians.setStyleSheet("font-size: 24px; font-weight: bold; color: #0088FF; background-color: #333; padding: 10px; border-radius: 5px;")
        counters_layout.addWidget(QLabel("PEATONES:"), 2, 0)
        counters_layout.addWidget(self.label_pedestrians, 2, 1)
        
        # Direction A->B
        self.label_direction_ab = QLabel("0")
        self.label_direction_ab.setStyleSheet("font-size: 20px; font-weight: bold; color: #FFAA00; background-color: #333; padding: 8px;")
        counters_layout.addWidget(QLabel("A→B:"), 3, 0)
        counters_layout.addWidget(self.label_direction_ab, 3, 1)
        
        # Direction B->A
        self.label_direction_ba = QLabel("0")
        self.label_direction_ba.setStyleSheet("font-size: 20px; font-weight: bold; color: #AA00FF; background-color: #333; padding: 8px;")
        counters_layout.addWidget(QLabel("B→A:"), 4, 0)
        counters_layout.addWidget(self.label_direction_ba, 4, 1)
        
        layout.addLayout(counters_layout)
        
        # Performance metrics
        layout.addWidget(QLabel("--- RENDIMIENTO ---"))
        
        metrics_layout = QGridLayout()
        
        self.label_camera_fps = QLabel("0 FPS")
        metrics_layout.addWidget(QLabel("FPS Cámara:"), 0, 0)
        metrics_layout.addWidget(self.label_camera_fps, 0, 1)
        
        self.label_inference_fps = QLabel("0 FPS")
        metrics_layout.addWidget(QLabel("FPS Inferencia:"), 1, 0)
        metrics_layout.addWidget(self.label_inference_fps, 1, 1)
        
        self.label_display_fps = QLabel("0 FPS")
        metrics_layout.addWidget(QLabel("FPS Pantalla:"), 2, 0)
        metrics_layout.addWidget(self.label_display_fps, 2, 1)
        
        self.label_latency = QLabel("0 ms")
        metrics_layout.addWidget(QLabel("Latencia:"), 3, 0)
        metrics_layout.addWidget(self.label_latency, 3, 1)
        
        self.label_session_time = QLabel("00:00:00")
        metrics_layout.addWidget(QLabel("Tiempo Sesión:"), 4, 0)
        metrics_layout.addWidget(self.label_session_time, 4, 1)

        self.label_quality = QLabel("SIN DATOS")
        self.label_quality.setStyleSheet("font-weight: bold; color: #aaaaaa; padding: 6px;")
        metrics_layout.addWidget(QLabel("Calidad medición:"), 5, 0)
        metrics_layout.addWidget(self.label_quality, 5, 1)
        
        layout.addLayout(metrics_layout)
        
        # Intervals
        layout.addWidget(QLabel("--- INTERVALOS ---"))
        
        intervals_layout = QGridLayout()
        
        self.labels_intervals = {}
        for i, minutes in enumerate([1, 5, 15, 60]):
            label_name = QLabel(f"{minutes}m:")
            label_value = QLabel("0")
            label_value.setStyleSheet("font-weight: bold;")
            
            intervals_layout.addWidget(label_name, i, 0)
            intervals_layout.addWidget(label_value, i, 1)
            
            self.labels_intervals[minutes] = label_value
        
        layout.addLayout(intervals_layout)
        layout.addStretch()
    
    def init_events_tab(self):
        """Initialize events tab"""
        layout = QVBoxLayout(self.tab_events)
        
        # Events table
        self.events_table = QTableWidget()
        self.events_table.setColumnCount(6)
        self.events_table.setHorizontalHeaderLabels([
            'Hora', 'Clase', 'ID Track', 'Dirección', 'Confianza', 'Línea'
        ])
        self.events_table.setColumnWidth(0, 120)
        self.events_table.setColumnWidth(1, 80)
        self.events_table.setColumnWidth(2, 70)
        self.events_table.setColumnWidth(3, 70)
        self.events_table.setColumnWidth(4, 80)
        
        layout.addWidget(self.events_table)
        
        # Clear button
        btn_clear = QPushButton("Limpiar historial")
        btn_clear.clicked.connect(self.clear_events)
        layout.addWidget(btn_clear)
    
    def init_export_tab(self):
        """Initialize export tab"""
        layout = QVBoxLayout(self.tab_export)
        
        layout.addWidget(QLabel("EXPORTAR DATOS DE LA SESIÓN"))
        layout.addWidget(QLabel(f"ID Sesión: {self.session_id}"))
        
        # Export options
        options_layout = QVBoxLayout()
        
        self.checkbox_csv = QCheckBox("CSV")
        self.checkbox_csv.setChecked(True)
        options_layout.addWidget(self.checkbox_csv)
        
        self.checkbox_excel = QCheckBox("Excel (XLSX)")
        self.checkbox_excel.setChecked(True)
        options_layout.addWidget(self.checkbox_excel)
        
        layout.addLayout(options_layout)
        
        # Export button
        btn_export = QPushButton("📤 EXPORTAR")
        btn_export.setStyleSheet("background-color: blue; color: white; font-weight: bold; padding: 10px;")
        btn_export.clicked.connect(self.export_data)
        layout.addWidget(btn_export)
        
        # Export status
        self.label_export_status = QLabel("Listo para exportar")
        layout.addWidget(self.label_export_status)
        
        layout.addStretch()
    
    def start_capture(self):
        """Start video capture and processing"""
        if self.is_running:
            return
        
        # Start video capture
        if not self.video_capture.start():
            QMessageBox.critical(self, "Error", "No se pudo conectar a la cámara")
            return
        
        # Start processing thread
        self.processing_thread = ProcessingThread(
            self.video_capture,
            self.detector,
            self.track_manager,
            self.line_detector,
            self.analytics_engine,
            self.database,
            self.session_id,
            self.crossing_engine
        )
        
        self.processing_thread.frame_ready.connect(self.on_frame_ready)
        self.processing_thread.error_occurred.connect(self.on_processing_error)
        self.processing_thread.start()
        
        # Start display timer
        self.display_timer.start(33)  # ~30 FPS display
        
        self.is_running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
    
    def stop_capture(self):
        """Stop video capture and processing"""
        if not self.is_running:
            return
        
        # Stop timer
        self.display_timer.stop()
        
        # Stop processing thread
        if self.processing_thread:
            self.processing_thread.stop()
            self.processing_thread = None
        
        # Stop video capture
        self.video_capture.stop()
        
        self.is_running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
    
    @Slot(object)
    def on_frame_ready(self, result: Dict):
        """Handle frame from processing thread"""
        self.current_frame = result
        expires = time.monotonic() + 1.0
        for crossing in result.get('crossing_events', []):
            self.crossing_badges.append((expires, crossing))
        self.crossing_badges = [item for item in self.crossing_badges if item[0] > time.monotonic()]
    
    @Slot(str)
    def on_processing_error(self, error: str):
        """Handle processing error"""
        print(f"Processing error: {error}")
        QMessageBox.warning(self, "Error de Procesamiento", f"Error: {error}")
    
    def update_display(self):
        """Update display with current frame"""
        if not self.current_frame:
            connected, error = self.video_capture.get_connection_status()
            if self.is_running and not connected:
                self.video_label.setText(
                    "CÁMARA DESCONECTADA\nReconectando automáticamente…" +
                    (f"\n{error}" if error else "")
                )
                self.video_label.setStyleSheet(
                    "background-color: #10151c; color: #ff5d73; "
                    "font-size: 22px; font-weight: bold; border: 2px solid #394657;"
                )
            return
        
        frame = self.current_frame['frame'].copy()
        detections = self.current_frame.get('detections', [])
        tracks = self.current_frame.get('tracks', {})
        camera_fps = self.current_frame.get('camera_fps', 0)
        inference_time = self.current_frame.get('inference_time', 0)
        display_fps = self.current_frame.get('fps', 0)
        
        # Draw counting line
        lines = self.line_detector.get_lines()
        for line_id, line in lines.items():
            if line.enabled:
                # Convert normalized to pixel coordinates if needed
                h, w = frame.shape[:2]
                x1 = int(line.norm_x1 * w)
                y1 = int(line.norm_y1 * h)
                x2 = int(line.norm_x2 * w)
                y2 = int(line.norm_y2 * h)
                
                utils.draw_counting_line(frame, x1, y1, x2, y2, line.name)

        # Non-blocking one-second confirmation badges for registered events.
        now_mono = time.monotonic()
        self.crossing_badges = [item for item in self.crossing_badges if item[0] > now_mono]
        for _, event in self.crossing_badges:
            point = (int(event.x), int(event.y))
            cv2.circle(frame, point, 9, (0, 255, 140), -1)
            badge = f"{event.class_name.upper()} #{event.tracker_id}  {event.direction_name}  REGISTRADO"
            cv2.putText(frame, badge, (max(8, point[0]-100), max(25, point[1]-18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 140), 2, cv2.LINE_AA)
        
        # Draw detections
        for detection in detections:
            utils.draw_bounding_box(
                frame, detection['bbox'],
                detection['id'],
                detection['class_label'],
                detection['confidence'],
                show_confidence=self.config.display.show_confidence,
                show_track_id=self.config.display.show_track_id
            )
        
        # Draw trajectories
        for track_id, track in tracks.items():
            trajectory = track.get_trajectory_points()
            if len(trajectory) > 1:
                utils.draw_trajectory(frame, trajectory)
        
        # Draw info panel
        info = {
            'Cámara': f"{camera_fps:.1f} FPS",
            'Inferencia': f"{inference_time*1000:.1f} ms",
            'Pantalla': f"{display_fps:.1f} FPS",
            'Tracks': len(tracks)
        }
        utils.draw_info_panel(frame, info, position="top-right")
        
        # Convert frame to QPixmap
        h, w, ch = frame.shape
        bytes_per_line = 3 * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888).rgbSwapped()
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale to fit label
        scaled_pixmap = pixmap.scaledToWidth(self.video_label.width(), Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #394657;")
        
        # Update stats
        self.update_stats()
        
        # Update display FPS
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        self.last_frame_time = current_time
    
    def update_stats(self):
        """Update statistics display"""
        active_counts = {}
        if self.current_frame:
            for track in self.current_frame.get('tracks', {}).values():
                name = getattr(track, 'class_name', 'unknown')
                active_counts[name] = active_counts.get(name, 0) + 1
        self.label_active_objects.setText(
            f"Persona: {active_counts.get('person', 0)}   "
            f"Carro: {active_counts.get('car', 0)}   "
            f"Moto: {active_counts.get('motorcycle', 0)}   "
            f"Otros: {sum(v for k, v in active_counts.items() if k not in ('person','car','motorcycle'))}"
        )
        # Get totals
        total = self.crossing_engine.get_count(1)
        
        # Get by class
        class_dist = self.crossing_engine.get_count_by_class(1)
        vehicles = sum(c for k, c in class_dist.items() if k.lower() not in ('persona', 'person'))
        pedestrians = class_dist.get('Persona', 0) + class_dist.get('person', 0)
        
        # Get by direction
        dir_ab = self.crossing_engine.get_count(1, 'A->B')
        dir_ba = self.crossing_engine.get_count(1, 'B->A')
        
        # Update labels
        self.label_total.setText(str(total))
        self.label_vehicles.setText(str(vehicles))
        self.label_pedestrians.setText(str(pedestrians))
        self.label_direction_ab.setText(str(dir_ab))
        self.label_direction_ba.setText(str(dir_ba))
        
        # Update FPS labels
        if self.current_frame:
            self.label_camera_fps.setText(f"{self.current_frame.get('camera_fps', 0):.1f} FPS")
            # FIX: inference_time está en segundos, no milisegundos
            inference_time_sec = max(self.current_frame.get('inference_time', 1), 0.001)
            inference_fps = 1.0 / inference_time_sec if inference_time_sec > 0 else 0
            self.label_inference_fps.setText(f"{inference_fps:.1f} FPS")
            self.label_display_fps.setText(f"{self.current_frame.get('fps', 0):.1f} FPS")
            self.label_latency.setText(f"{self.current_frame.get('inference_time', 0)*1000:.1f} ms")
            quality_report = self.current_frame.get('quality')
            if quality_report:
                quality_name = quality_report['quality_level']
                quality_colors = {
                    'EXCELENTE': '#2ecc71', 'BUENO': '#a3cb38',
                    'LIMITADO': '#f39c12', 'NO CONFIABLE': '#e74c3c'
                }
                self.label_quality.setText(quality_name)
                self.label_quality.setStyleSheet(
                    f"font-weight: bold; color: {quality_colors.get(quality_name, '#aaaaaa')}; padding: 6px;"
                )
        
        # Update session time
        if self.is_running:
            elapsed = self.analytics_engine.get_session_duration()
            self.label_session_time.setText(utils.format_duration(int(elapsed.total_seconds())))
        
        # Update intervals
        for minutes, label in self.labels_intervals.items():
            volume = self.analytics_engine.get_volume_for_interval(minutes)
            label.setText(str(volume))
        
        # Update events table
        self.update_events_table()
    
    def update_events_table(self):
        """Update events table"""
        events = self.crossing_engine.get_events(last_n=20)
        
        self.events_table.setRowCount(len(events))
        
        for i, event in enumerate(reversed(events)):
            # Time
            timestamp = datetime.fromisoformat(event.timestamp)
            self.events_table.setItem(i, 0, QTableWidgetItem(timestamp.strftime("%H:%M:%S.%f")[:-3]))
            
            # Class
            self.events_table.setItem(i, 1, QTableWidgetItem(event.class_label))
            
            # Track ID
            self.events_table.setItem(i, 2, QTableWidgetItem(str(event.track_id)))
            
            # Direction
            self.events_table.setItem(i, 3, QTableWidgetItem(event.direction))
            
            # Confidence
            self.events_table.setItem(i, 4, QTableWidgetItem(f"{event.confidence:.2f}"))
            
            # Line
            line = self.crossing_engine.get_line(event.line_id)
            line_name = line.name if line else "?"
            self.events_table.setItem(i, 5, QTableWidgetItem(line_name))
    
    def clear_events(self):
        """Clear events history"""
        reply = QMessageBox.question(
            self, "Confirmar",
            "¿Desea borrar el historial de eventos?"
        )
        
        if reply == QMessageBox.Yes:
            self.crossing_engine.clear_history()
            self.events_table.setRowCount(0)
    
    def open_settings(self):
        """Open settings dialog"""
        dialog = SettingsDialog(self, self.config_manager)
        
        if dialog.exec() == QDialog.Accepted:
            settings = dialog.get_settings()
            
            # Update configuration
            self.config.camera.source = settings['camera_url']
            self.config.camera.source_type = settings['camera_type']
            self.config.detection.confidence = settings['confidence']
            self.config.detection.imgsz = settings['imgsz']
            self.config.detection.device = settings['device']
            self.config.performance_mode = settings['performance_mode']
            
            self.config_manager.save()
            
            # Apply new settings
            self.detector.update_confidence(settings['confidence'])
            self.detector.update_imgsz(settings['imgsz'])
            
            # If running, need to restart
            if self.is_running:
                QMessageBox.information(
                    self, "Configuración",
                    "Deberá reiniciar la captura para aplicar los cambios"
                )
    
    def open_line_editor(self):
        """Open line editor dialog"""
        line = self.line_detector.get_line(1)  # Assuming line_id = 1
        
        frame_shape = self.current_frame['frame'].shape if self.current_frame else (480, 640, 3)
        dialog = LineEditorDialog(self, line, frame_shape)
        
        if dialog.exec() == QDialog.Accepted:
            line_data = dialog.get_line_data()
            
            # Update line
            frame_height, frame_width = frame_shape[:2]
            self.line_detector.update_line_from_normalized(
                1,
                line_data['norm_x1'],
                line_data['norm_y1'],
                line_data['norm_x2'],
                line_data['norm_y2'],
                frame_width,
                frame_height
            )
            
            # Update name
            line.name = line_data['name']
            line.direction_a = line_data['direction_a']
            line.direction_b = line_data['direction_b']
            
            # Update config
            self.config.line.x1 = line_data['norm_x1']
            self.config.line.y1 = line_data['norm_y1']
            self.config.line.x2 = line_data['norm_x2']
            self.config.line.y2 = line_data['norm_y2']
            self.config.line.name = line_data['name']
            
            self.config_manager.save()
    
    def export_data(self):
        """Export session data"""
        formats = []
        
        if self.checkbox_csv.isChecked():
            formats.append('csv')
        if self.checkbox_excel.isChecked():
            formats.append('excel')
        
        if not formats:
            QMessageBox.warning(self, "Error", "Seleccione al menos un formato")
            return
        
        try:
            # Get session data
            session_data = self.database.export_session_to_dict(self.session_id)
            
            # Export
            results = self.exporter.export_session_complete(session_data, self.session_id, formats)
            
            if results:
                message = "Archivos exportados:\n"
                for fmt, path in results.items():
                    message += f"\n{fmt.upper()}: {path}"
                
                self.label_export_status.setText(f"Exportado en {', '.join(results.keys())}")
                QMessageBox.information(self, "Exportación", message)
            else:
                self.label_export_status.setText("Error en la exportación")
                QMessageBox.warning(self, "Error", "No se pudo exportar los datos")
        
        except Exception as e:
            self.label_export_status.setText(f"Error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error al exportar: {str(e)}")
    
    def closeEvent(self, event):
        """Handle application close"""
        if self.is_running:
            self.stop_capture()
        
        # End session
        self.database.end_session(self.session_id)
        
        event.accept()


def main():
    """Main application entry point"""
    from PySide6.QtWidgets import QApplication
    
    app = QApplication([])
    window = TrafficDetectionApp()
    window.show()
    
    import sys
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
