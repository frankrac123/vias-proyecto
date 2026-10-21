"""
YOLO Detection module
Wraps Ultralytics YOLO with configurable parameters
"""
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional
import torch


class TrafficDetector:
    """YOLO-based traffic object detector"""
    
    # COCO class mapping for traffic analysis
    VALID_CLASSES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }
    
    CLASS_LABELS_ES = {
        0: "Persona",
        1: "Bicicleta",
        2: "Carro",
        3: "Moto",
        5: "Bus",
        7: "Camión"
    }
    
    def __init__(self, model_path: str = "yolo11n.pt", 
                 confidence: float = 0.30,
                 imgsz: int = 416,
                 device: str = "auto"):
        """
        Initialize detector
        
        Args:
            model_path: Path to YOLO model weights
            confidence: Confidence threshold for detections
            imgsz: Input image size for model
            device: Device to use ('cpu', 'cuda', '0', '1', or 'auto')
        """
        self.model_path = model_path
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = self._resolve_device(device)
        
        self.model = None
        self.is_loaded = False
        self.inference_time = 0.0
        
        self._load_model()
    
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device"""
        if device.lower() == "auto":
            # Auto-detect CUDA
            return "0" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            # Move to device
            self.model.to(self.device)
            self.is_loaded = True
            print(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_loaded = False
            raise
    
    def detect(self, frame: np.ndarray, 
               classes: Optional[List[int]] = None,
               tracker_conf: str = "bytetrack.yaml") -> Dict:
        """
        Detect and track objects in frame
        
        Args:
            frame: Input frame (BGR format)
            classes: List of COCO class IDs to detect
            tracker_conf: Tracker configuration file
        
        Returns:
            Dict with detection results
        """
        if not self.is_loaded or self.model is None:
            return {'success': False, 'error': 'Model not loaded'}
        
        if classes is None:
            classes = list(self.VALID_CLASSES.keys())
        
        try:
            results = self.model.track(
                frame,
                persist=True,
                classes=classes,
                conf=self.confidence,
                imgsz=self.imgsz,
                device=self.device,
                tracker=tracker_conf,
                verbose=False
            )
            
            detection_data = self._parse_results(results, classes)
            self.inference_time = results[0].speed.get('inference', 0) / 1000.0  # Convert to seconds
            
            return {
                'success': True,
                'detections': detection_data,
                'frame_shape': frame.shape,
                'inference_time': self.inference_time
            }
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _parse_results(self, results, classes: List[int]) -> List[Dict]:
        """
        Parse YOLO results into detection list
        
        Returns:
            List of detection dicts with keys:
            - id: Track ID
            - class_id: COCO class ID
            - class_name: Class name (English)
            - class_label: Class label (Spanish)
            - confidence: Detection confidence
            - bbox: [x1, y1, x2, y2] coordinates
            - center_x: Center X coordinate
            - center_y: Center Y coordinate
        """
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.int().cpu().tolist()
                class_ids = result.boxes.cls.int().cpu().tolist()
                confidences = result.boxes.conf.cpu().numpy()
                
                for box, obj_id, class_id, conf in zip(boxes, ids, class_ids, confidences):
                    if class_id in classes and class_id in self.VALID_CLASSES:
                        x1, y1, x2, y2 = map(float, box)
                        center_x = (x1 + x2) / 2.0
                        center_y = (y1 + y2) / 2.0
                        
                        detection = {
                            'id': int(obj_id),
                            'class_id': int(class_id),
                            'class_name': self.VALID_CLASSES[class_id],
                            'class_label': self.CLASS_LABELS_ES.get(class_id, 'Unknown'),
                            'confidence': float(conf),
                            'bbox': [x1, y1, x2, y2],
                            'center_x': center_x,
                            'center_y': center_y,
                            'width': x2 - x1,
                            'height': y2 - y1
                        }
                        detections.append(detection)
        
        return detections
    
    def update_confidence(self, confidence: float):
        """Update detection confidence threshold"""
        self.confidence = max(0.0, min(1.0, confidence))
    
    def update_imgsz(self, imgsz: int):
        """Update input image size"""
        valid_sizes = [320, 416, 512, 640, 1280]
        if imgsz not in valid_sizes:
            print(f"Warning: {imgsz} not in standard sizes. Using closest: {min(valid_sizes, key=lambda x: abs(x - imgsz))}")
        self.imgsz = imgsz
    
    def get_inference_time(self) -> float:
        """Get last inference time in seconds"""
        return self.inference_time
    
    def get_device(self) -> str:
        """Get current device"""
        return self.device
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        if not self.is_loaded:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'model_path': self.model_path,
            'device': self.device,
            'confidence': self.confidence,
            'imgsz': self.imgsz,
            'supported_classes': self.VALID_CLASSES
        }
