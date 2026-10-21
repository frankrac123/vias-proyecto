"""
Utility functions for traffic detection system
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path


def draw_bounding_box(frame: np.ndarray, 
                      bbox: List[float],
                      track_id: int,
                      class_label: str,
                      confidence: float,
                      show_confidence: bool = True,
                      show_track_id: bool = True,
                      line_width: int = 2) -> np.ndarray:
    """Draw bounding box on frame"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), line_width)
    
    # Draw center point
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
    
    # Prepare label
    label_parts = [class_label]
    if show_track_id:
        label_parts.append(f"ID:{track_id}")
    if show_confidence:
        label_parts.append(f"{confidence:.2f}")
    
    label = " | ".join(label_parts)
    
    # Draw label background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    text_x = x1
    text_y = max(y1 - 5, text_size[1] + 5)
    
    cv2.rectangle(frame, 
                  (text_x, text_y - text_size[1] - 5),
                  (text_x + text_size[0] + 5, text_y + 5),
                  (0, 255, 0), -1)
    
    # Draw label text
    cv2.putText(frame, label, (text_x + 2, text_y - 2),
                font, font_scale, (0, 0, 0), thickness)
    
    return frame


def draw_trajectory(frame: np.ndarray,
                   trajectory: List[Tuple[float, float]],
                   color: Tuple[int, int, int] = (0, 255, 255),
                   thickness: int = 2) -> np.ndarray:
    """Draw trajectory line on frame"""
    if len(trajectory) < 2:
        return frame
    
    # Draw polyline
    points = np.array(trajectory, dtype=np.int32)
    cv2.polylines(frame, [points], False, color, thickness)
    
    return frame


def draw_counting_line(frame: np.ndarray,
                      x1: float, y1: float, x2: float, y2: float,
                      name: str = "Línea de Conteo",
                      color: Tuple[int, int, int] = (0, 255, 255),
                      thickness: int = 3) -> np.ndarray:
    """Draw counting line on frame"""
    pt1 = (int(x1), int(y1))
    pt2 = (int(x2), int(y2))
    
    cv2.line(frame, pt1, pt2, color, thickness)
    
    # Draw circles at endpoints
    cv2.circle(frame, pt1, 6, color, -1)
    cv2.circle(frame, pt2, 6, color, -1)
    
    # Draw label
    mid_x = (pt1[0] + pt2[0]) // 2
    mid_y = (pt1[1] + pt2[1]) // 2
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness_text = 2
    
    text_size = cv2.getTextSize(name, font, font_scale, thickness_text)[0]
    text_x = mid_x - text_size[0] // 2
    text_y = mid_y - 10
    
    # Background for text
    cv2.rectangle(frame,
                  (text_x - 5, text_y - text_size[1] - 5),
                  (text_x + text_size[0] + 5, text_y + 5),
                  (0, 0, 0), -1)
    
    cv2.putText(frame, name, (text_x, text_y),
                font, font_scale, color, thickness_text)
    
    return frame


def draw_info_panel(frame: np.ndarray,
                   info_dict: dict,
                   position: str = "top-left") -> np.ndarray:
    """
    Draw information panel on frame
    
    Args:
        frame: Input frame
        info_dict: Dictionary of info to display
        position: 'top-left', 'top-right', 'bottom-left', 'bottom-right'
    
    Returns:
        Frame with info panel
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_spacing = 25
    
    # Prepare text
    lines = []
    for key, value in info_dict.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.1f}")
        else:
            lines.append(f"{key}: {value}")
    
    if not lines:
        return frame
    
    # Calculate panel size
    max_width = 0
    for line in lines:
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        max_width = max(max_width, text_size[0])
    
    panel_width = max_width + 20
    panel_height = len(lines) * line_spacing + 10
    
    # Determine position
    if position == "top-left":
        x, y = 10, 10
    elif position == "top-right":
        x, y = frame.shape[1] - panel_width - 10, 10
    elif position == "bottom-left":
        x, y = 10, frame.shape[0] - panel_height - 10
    else:  # bottom-right
        x, y = frame.shape[1] - panel_width - 10, frame.shape[0] - panel_height - 10
    
    # Draw background
    cv2.rectangle(frame,
                  (x, y),
                  (x + panel_width, y + panel_height),
                  (0, 0, 0), -1)
    
    # Draw border
    cv2.rectangle(frame,
                  (x, y),
                  (x + panel_width, y + panel_height),
                  (0, 255, 255), 2)
    
    # Draw text
    for i, line in enumerate(lines):
        text_y = y + 10 + i * line_spacing + 15
        cv2.putText(frame, line, (x + 10, text_y),
                    font, font_scale, (0, 255, 255), thickness)
    
    return frame


def normalize_coordinates(x: float, y: float,
                         frame_width: int, frame_height: int) -> Tuple[float, float]:
    """Convert pixel coordinates to normalized (0-1) coordinates"""
    return (x / frame_width, y / frame_height)


def denormalize_coordinates(norm_x: float, norm_y: float,
                            frame_width: int, frame_height: int) -> Tuple[float, float]:
    """Convert normalized coordinates to pixel coordinates"""
    return (norm_x * frame_width, norm_y * frame_height)


def ensure_path_exists(path: str) -> Path:
    """Ensure directory path exists"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_fps_string(fps: float) -> str:
    """Format FPS value as string"""
    if fps > 0:
        return f"{fps:.1f} FPS"
    return "0 FPS"


def format_duration(seconds: int) -> str:
    """Format duration from seconds to HH:MM:SS"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_color_by_class(class_label: str) -> Tuple[int, int, int]:
    """Get BGR color for class label"""
    colors = {
        'Persona': (255, 0, 0),      # Blue
        'Bicicleta': (0, 255, 255),  # Yellow
        'Carro': (0, 255, 0),        # Green
        'Moto': (0, 165, 255),       # Orange
        'Bus': (255, 0, 255),        # Magenta
        'Camión': (255, 255, 0)      # Cyan
    }
    return colors.get(class_label, (128, 128, 128))  # Default gray


def smooth_trajectory(trajectory: List[Tuple[float, float]], 
                     kernel_size: int = 3) -> List[Tuple[float, float]]:
    """Smooth trajectory points using moving average"""
    if len(trajectory) < kernel_size:
        return trajectory
    
    smoothed = []
    half_kernel = kernel_size // 2
    
    for i in range(len(trajectory)):
        start = max(0, i - half_kernel)
        end = min(len(trajectory), i + half_kernel + 1)
        
        window = trajectory[start:end]
        avg_x = sum(p[0] for p in window) / len(window)
        avg_y = sum(p[1] for p in window) / len(window)
        
        smoothed.append((avg_x, avg_y))
    
    return smoothed


def get_frame_shape_info(frame: np.ndarray) -> dict:
    """Get information about frame shape"""
    if len(frame.shape) == 3:
        height, width, channels = frame.shape
    else:
        height, width = frame.shape
        channels = 1
    
    return {
        'height': height,
        'width': width,
        'channels': channels,
        'resolution': f"{width}x{height}"
    }
