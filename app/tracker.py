"""
Object Tracking module
Manages track history and trajectory data
"""
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import time


@dataclass
class TrackPoint:
    """Single point in object trajectory"""
    x: float
    y: float
    timestamp: float
    frame_id: int


@dataclass
class Track:
    """Object track with history"""
    track_id: int
    class_id: int
    class_name: str
    class_label: str
    
    trajectory: List[TrackPoint] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    confidence: float = 0.0
    
    max_trajectory_length: int = 50
    
    def add_point(self, x: float, y: float, frame_id: int):
        """Add point to trajectory"""
        point = TrackPoint(x, y, time.time(), frame_id)
        self.trajectory.append(point)
        
        # Keep only recent trajectory points
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory.pop(0)
        
        self.last_seen = time.time()
    
    def get_last_position(self) -> Optional[Tuple[float, float]]:
        """Get last known position"""
        if self.trajectory:
            last = self.trajectory[-1]
            return (last.x, last.y)
        return None
    
    def get_previous_position(self) -> Optional[Tuple[float, float]]:
        """Get second-to-last position"""
        if len(self.trajectory) >= 2:
            prev = self.trajectory[-2]
            return (prev.x, prev.y)
        return None
    
    def get_trajectory_points(self) -> List[Tuple[float, float]]:
        """Get all trajectory points as list of (x, y)"""
        return [(p.x, p.y) for p in self.trajectory]
    
    def time_since_last_seen(self) -> float:
        """Get seconds since last detection"""
        return time.time() - self.last_seen
    
    def get_age(self) -> float:
        """Get track age in seconds"""
        return self.last_seen - self.first_seen


class TrackManager:
    """Manages all active tracks"""
    
    def __init__(self, max_age: float = 5.0, max_trajectory_length: int = 50):
        """
        Initialize track manager
        
        Args:
            max_age: Maximum time to keep track after last detection (seconds)
            max_trajectory_length: Maximum trajectory points to store per track
        """
        self.tracks: Dict[int, Track] = {}
        self.max_age = max_age
        self.max_trajectory_length = max_trajectory_length
        self.frame_count = 0
        
        # Track history for avoiding double counting
        self.track_line_crossings: Dict[Tuple[int, int], int] = defaultdict(int)  # (track_id, line_id) -> count
    
    def update(self, detections: List[Dict], frame_id: int):
        """
        Update tracks with new detections
        
        Args:
            detections: List of detection dicts with 'id', 'center_x', 'center_y', 'class_id', 'class_name', etc.
            frame_id: Current frame number
        """
        self.frame_count = frame_id
        detected_ids = set()
        
        # Update existing tracks and create new ones
        for detection in detections:
            track_id = detection['id']
            detected_ids.add(track_id)
            
            if track_id in self.tracks:
                # Update existing track
                track = self.tracks[track_id]
                track.add_point(detection['center_x'], detection['center_y'], frame_id)
                track.confidence = detection['confidence']
            else:
                # Create new track
                track = Track(
                    track_id=track_id,
                    class_id=detection['class_id'],
                    class_name=detection['class_name'],
                    class_label=detection['class_label'],
                    confidence=detection['confidence'],
                    max_trajectory_length=self.max_trajectory_length
                )
                track.add_point(detection['center_x'], detection['center_y'], frame_id)
                self.tracks[track_id] = track
        
        # Remove old tracks
        expired_ids = []
        for track_id, track in self.tracks.items():
            if track_id not in detected_ids and track.time_since_last_seen() > self.max_age:
                expired_ids.append(track_id)
        
        for track_id in expired_ids:
            del self.tracks[track_id]
    
    def get_tracks(self) -> Dict[int, Track]:
        """Get all active tracks"""
        return self.tracks.copy()
    
    def get_track(self, track_id: int) -> Optional[Track]:
        """Get specific track"""
        return self.tracks.get(track_id)
    
    def track_crossed_line(self, track_id: int, line_id: int) -> bool:
        """
        Record that a track crossed a line
        
        Returns:
            True if this is the first crossing of this line by this track
        """
        key = (track_id, line_id)
        if self.track_line_crossings[key] == 0:
            self.track_line_crossings[key] += 1
            return True
        return False
    
    def get_track_crossings(self, track_id: int, line_id: int) -> int:
        """Get number of times a track has crossed a specific line"""
        return self.track_line_crossings[(track_id, line_id)]
    
    def clear_old_crossings(self, max_frames: int = 1000):
        """Clear old crossing records to free memory"""
        current_ids = set(self.tracks.keys())
        to_delete = []
        
        for (track_id, line_id) in self.track_line_crossings.keys():
            if track_id not in current_ids:
                to_delete.append((track_id, line_id))
        
        for key in to_delete:
            del self.track_line_crossings[key]
    
    def get_active_count(self) -> int:
        """Get number of active tracks"""
        return len(self.tracks)
    
    def get_total_tracks_processed(self) -> int:
        """Get total number of unique tracks ever seen"""
        return len(self.track_line_crossings)
    
    def reset(self):
        """Reset all tracks"""
        self.tracks.clear()
        self.track_line_crossings.clear()
        self.frame_count = 0
