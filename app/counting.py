"""
Robust line-based counting using intersection geometry
Handles high-speed objects, diagonal lines, and various scenarios
"""
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class CountingLine:
    """Line for counting object crossings"""
    line_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    name: str = "Línea de Conteo"
    enabled: bool = True
    direction_a: str = "A"
    direction_b: str = "B"
    
    # Normalized coordinates for different resolutions
    norm_x1: float = 0.0
    norm_y1: float = 0.0
    norm_x2: float = 1.0
    norm_y2: float = 1.0
    
    def update_from_normalized(self, width: int, height: int):
        """Calculate pixel coordinates from normalized coordinates"""
        self.x1 = self.norm_x1 * width
        self.y1 = self.norm_y1 * height
        self.x2 = self.norm_x2 * width
        self.y2 = self.norm_y2 * height
    
    def update_normalized(self, width: int, height: int):
        """Calculate normalized coordinates from pixel coordinates"""
        self.norm_x1 = self.x1 / width if width > 0 else 0
        self.norm_y1 = self.y1 / height if height > 0 else 0
        self.norm_x2 = self.x2 / width if width > 0 else 0
        self.norm_y2 = self.y2 / height if height > 0 else 0
    
    def get_direction_vector(self) -> Tuple[float, float]:
        """Get normalized direction vector of the line"""
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        length = (dx**2 + dy**2)**0.5
        
        if length == 0:
            return (0, 0)
        
        return (dx / length, dy / length)
    
    def get_perpendicular_vector(self) -> Tuple[float, float]:
        """Get perpendicular vector to the line"""
        dx, dy = self.get_direction_vector()
        return (-dy, dx)


@dataclass
class CrossingEvent:
    """Record of a crossing event"""
    track_id: int
    class_id: int
    class_label: str
    timestamp: float
    frame_id: int
    line_id: int
    direction: str  # 'A->B' or 'B->A'
    position_x: float
    position_y: float
    confidence: float


class LineIntersectionDetector:
    """
    Detects line crossings using robust geometric intersection
    
    Handles:
    - High-speed objects (may skip over line in single frame)
    - Diagonal, horizontal, and vertical lines
    - Avoiding double-counting via track-line pair tracking
    """
    
    def __init__(self):
        self.lines: Dict[int, CountingLine] = {}
        self.line_counter = 0
        
        # Track which line was last detected for each track
        self.track_line_state: Dict[int, Tuple[int, float]] = {}  # track_id -> (side, frame_count)
        self.crossing_events: List[CrossingEvent] = []
        self.crossing_counts: Dict[Tuple[int, str], int] = defaultdict(int)  # (line_id, direction) -> count
    
    def add_line(self, x1: float, y1: float, x2: float, y2: float, 
                 name: str = "Línea", norm_coords: bool = True,
                 direction_a: str = "A", direction_b: str = "B") -> int:
        """
        Add a counting line
        
        Args:
            x1, y1, x2, y2: Line endpoints
            name: Line name
            norm_coords: Whether coordinates are normalized (0-1) or pixel coordinates
        
        Returns:
            Line ID
        """
        self.line_counter += 1
        line = CountingLine(
            line_id=self.line_counter,
            x1=x1, y1=y1, x2=x2, y2=y2,
            name=name, direction_a=direction_a, direction_b=direction_b,
            norm_x1=x1 if norm_coords else 0,
            norm_y1=y1 if norm_coords else 0,
            norm_x2=x2 if norm_coords else 1,
            norm_y2=y2 if norm_coords else 1
        )
        
        self.lines[self.line_counter] = line
        return self.line_counter
    
    def update_line_from_pixels(self, line_id: int, x1: float, y1: float, x2: float, y2: float,
                                width: int, height: int):
        """Update line with pixel coordinates and update normalized coords"""
        if line_id in self.lines:
            line = self.lines[line_id]
            line.x1, line.y1, line.x2, line.y2 = x1, y1, x2, y2
            line.update_normalized(width, height)
    
    def update_line_from_normalized(self, line_id: int, x1: float, y1: float, x2: float, y2: float,
                                   width: int, height: int):
        """Update line with normalized coordinates and update pixel coords"""
        if line_id in self.lines:
            line = self.lines[line_id]
            line.norm_x1, line.norm_y1, line.norm_x2, line.norm_y2 = x1, y1, x2, y2
            line.update_from_normalized(width, height)
    
    def _ccw(self, p1: Tuple[float, float], p2: Tuple[float, float], 
             p3: Tuple[float, float]) -> bool:
        """Check counter-clockwise orientation"""
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])
    
    def _segments_intersect(self, p1: Tuple[float, float], p2: Tuple[float, float],
                           p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        """Check if two line segments intersect"""
        return self._ccw(p1, p3, p4) != self._ccw(p2, p3, p4) and \
               self._ccw(p1, p2, p3) != self._ccw(p1, p2, p4)
    
    def _point_to_line_side(self, point: Tuple[float, float], 
                           line_p1: Tuple[float, float],
                           line_p2: Tuple[float, float]) -> float:
        """
        Determine which side of line point is on using cross product
        Returns positive, negative, or zero
        """
        return (line_p2[0] - line_p1[0]) * (point[1] - line_p1[1]) - \
               (line_p2[1] - line_p1[1]) * (point[0] - line_p1[0])
    
    def _get_intersection_point(self, p1: Tuple[float, float], p2: Tuple[float, float],
                               p3: Tuple[float, float], p4: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Get intersection point of two line segments"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        if 0 <= t <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
        
        return None
    
    def check_crossing(self, track_id: int, prev_pos: Optional[Tuple[float, float]],
                      curr_pos: Tuple[float, float], 
                      track_data: Dict,
                      line_id: int) -> Optional[Tuple[str, Tuple[float, float]]]:
        """
        Check if track crossed line using trajectory segment
        
        Args:
            track_id: ID of track
            prev_pos: Previous position of track center
            curr_pos: Current position of track center
            track_data: Track data dict with 'class_id', 'class_label', 'confidence'
            line_id: ID of counting line
        
        Returns:
            Tuple of (direction, intersection_point) if crossing detected, else None
        """
        if line_id not in self.lines:
            return None
        
        line = self.lines[line_id]
        if not line.enabled:
            return None
        
        # If no previous position, can't determine crossing
        if prev_pos is None:
            return None
        
        # Check if trajectory segment intersects line
        if not self._segments_intersect(prev_pos, curr_pos, (line.x1, line.y1), (line.x2, line.y2)):
            return None
        
        # Get intersection point
        intersection = self._get_intersection_point(
            prev_pos, curr_pos,
            (line.x1, line.y1), (line.x2, line.y2)
        )
        
        # Determine direction using cross product
        prev_side = self._point_to_line_side(prev_pos, (line.x1, line.y1), (line.x2, line.y2))
        curr_side = self._point_to_line_side(curr_pos, (line.x1, line.y1), (line.x2, line.y2))
        
        # Check if sides changed (crossing occurred)
        if prev_side * curr_side < 0:  # Different signs = crossing
            direction = "A->B" if prev_side < 0 else "B->A"
            return (direction, intersection if intersection else curr_pos)
        
        return None
    
    def process_tracks(self, tracks: Dict, frame_id: int) -> List[CrossingEvent]:
        """
        Process all tracks and detect crossings
        
        Args:
            tracks: Dict of Track objects from TrackManager
            frame_id: Current frame ID
        
        Returns:
            List of new crossing events
        """
        events = []
        
        for line_id in self.lines:
            if not self.lines[line_id].enabled:
                continue
            
            for track_id, track in tracks.items():
                # Get current and previous positions
                curr_pos = track.get_last_position()
                prev_pos = track.get_previous_position()
                
                if curr_pos is None:
                    continue
                
                # Check for crossing
                crossing = self.check_crossing(
                    track_id, prev_pos, curr_pos,
                    {
                        'class_id': track.class_id,
                        'class_label': track.class_label,
                        'confidence': track.confidence
                    },
                    line_id
                )
                
                if crossing:
                    direction, intersection = crossing
                    
                    # Avoid double counting: check if this exact crossing was already recorded
                    key = (track_id, line_id)
                    if (track_id, line_id, direction) not in [(e.track_id, e.line_id, e.direction) 
                                                              for e in self.crossing_events[-100:]]:  # Check recent events
                        
                        event = CrossingEvent(
                            track_id=track_id,
                            class_id=track.class_id,
                            class_label=track.class_label,
                            timestamp=time.time(),
                            frame_id=frame_id,
                            line_id=line_id,
                            direction=direction,
                            position_x=intersection[0],
                            position_y=intersection[1],
                            confidence=track.confidence
                        )
                        
                        events.append(event)
                        self.crossing_events.append(event)
                        self.crossing_counts[(line_id, direction)] += 1
        
        return events
    
    def get_count(self, line_id: int, direction: Optional[str] = None) -> int:
        """Get crossing count for line"""
        if direction is None:
            return sum(count for (lid, _), count in self.crossing_counts.items() if lid == line_id)
        return self.crossing_counts.get((line_id, direction), 0)
    
    def get_count_by_class(self, line_id: int, direction: Optional[str] = None) -> Dict[str, int]:
        """Get crossing count by class"""
        result = defaultdict(int)
        
        for event in self.crossing_events:
            if event.line_id != line_id:
                continue
            if direction is not None and event.direction != direction:
                continue
            result[event.class_label] += 1
        
        return dict(result)
    
    def get_line(self, line_id: int) -> Optional[CountingLine]:
        """Get line by ID"""
        return self.lines.get(line_id)
    
    def get_lines(self) -> Dict[int, CountingLine]:
        """Get all lines"""
        return self.lines.copy()
    
    def clear_history(self):
        """Clear crossing history"""
        self.crossing_events.clear()
        self.crossing_counts.clear()
    
    def get_events(self, line_id: Optional[int] = None, 
                   last_n: int = 100) -> List[CrossingEvent]:
        """Get recent crossing events"""
        events = self.crossing_events[-last_n:]
        
        if line_id is not None:
            events = [e for e in events if e.line_id == line_id]
        
        return events
