"""
Real-time analytics engine for traffic flow analysis
Calculates volume, flow rates, and statistics
"""
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
import time


class AnalyticsEngine:
    """Calculates real-time traffic analytics"""
    
    def __init__(self, intervals: List[int] = None):
        """
        Initialize analytics engine
        
        Args:
            intervals: List of time intervals in minutes to track (default: [1, 5, 15, 60])
        """
        self.intervals = intervals or [1, 5, 15, 60]
        
        # Track events with timestamps
        self.events: deque = deque()  # All events with timestamp
        
        # Interval-based counts
        self.interval_counts: Dict[int, deque] = {
            i: deque() for i in self.intervals
        }  # (timestamp, count) tuples
        
        # Per-class counts
        self.class_counts: Dict[str, int] = defaultdict(int)
        
        # Direction counts
        self.direction_counts: Dict[str, int] = defaultdict(int)
        
        # Peak tracking
        self.peak_time: Optional[datetime] = None
        self.peak_volume: int = 0
        
        self.session_start_time = datetime.now()
    
    def add_event(self, event_data: Dict):
        """Add crossing event"""
        event_data['timestamp'] = time.time()
        self.events.append(event_data)
        
        # Update class count
        class_label = event_data.get('class_label', 'Unknown')
        self.class_counts[class_label] += 1
        
        # Update direction count
        direction = event_data.get('direction', 'Unknown')
        self.direction_counts[direction] += 1
    
    def get_volume_for_interval(self, minutes: int) -> int:
        """
        Get event count for the last N minutes
        
        Args:
            minutes: Time interval in minutes
        
        Returns:
            Number of events in interval
        """
        cutoff_time = time.time() - (minutes * 60)
        return sum(1 for event in self.events if event.get('timestamp', 0) > cutoff_time)
    
    def get_flow_rate(self, minutes: int) -> float:
        """
        Get flow rate (events per minute)
        
        Args:
            minutes: Time interval in minutes
        
        Returns:
            Flow rate in events/minute
        """
        volume = self.get_volume_for_interval(minutes)
        return volume / minutes if minutes > 0 else 0
    
    def get_flow_rate_per_hour(self, minutes: int) -> float:
        """Get flow rate in events per hour"""
        return self.get_flow_rate(minutes) * 60
    
    def get_class_distribution(self, minutes: Optional[int] = None) -> Dict[str, int]:
        """
        Get event count by class
        
        Args:
            minutes: If specified, only count events from last N minutes
        
        Returns:
            Dict with class labels and counts
        """
        if minutes is None:
            return dict(self.class_counts)
        
        cutoff_time = time.time() - (minutes * 60)
        distribution = defaultdict(int)
        
        for event in self.events:
            if event.get('timestamp', 0) > cutoff_time:
                distribution[event.get('class_label', 'Unknown')] += 1
        
        return dict(distribution)
    
    def get_direction_distribution(self, minutes: Optional[int] = None) -> Dict[str, int]:
        """Get event count by direction"""
        if minutes is None:
            return dict(self.direction_counts)
        
        cutoff_time = time.time() - (minutes * 60)
        distribution = defaultdict(int)
        
        for event in self.events:
            if event.get('timestamp', 0) > cutoff_time:
                distribution[event.get('direction', 'Unknown')] += 1
        
        return dict(distribution)
    
    def get_vehicle_count(self, minutes: Optional[int] = None) -> int:
        """Get count of vehicles (exclude pedestrians)"""
        vehicle_classes = ['Carro', 'Moto', 'Bus', 'Camión', 'Bicicleta']
        
        if minutes is None:
            return sum(count for cls, count in self.class_counts.items() 
                      if cls in vehicle_classes)
        
        cutoff_time = time.time() - (minutes * 60)
        count = 0
        for event in self.events:
            if event.get('timestamp', 0) > cutoff_time:
                if event.get('class_label') in vehicle_classes:
                    count += 1
        
        return count
    
    def get_pedestrian_count(self, minutes: Optional[int] = None) -> int:
        """Get count of pedestrians"""
        if minutes is None:
            return self.class_counts.get('Persona', 0)
        
        cutoff_time = time.time() - (minutes * 60)
        count = 0
        for event in self.events:
            if event.get('timestamp', 0) > cutoff_time:
                if event.get('class_label') == 'Persona':
                    count += 1
        
        return count
    
    def get_total_volume(self) -> int:
        """Get total volume for entire session"""
        return len(self.events)
    
    def get_session_duration(self) -> timedelta:
        """Get session duration"""
        return datetime.now() - self.session_start_time
    
    def get_average_flow_rate(self) -> float:
        """Get average flow rate for entire session (events per minute)"""
        duration_seconds = self.get_session_duration().total_seconds()
        if duration_seconds > 0:
            return (len(self.events) / duration_seconds) * 60
        return 0
    
    def get_statistics(self, minutes: int = 60) -> Dict:
        """
        Get comprehensive statistics
        
        Args:
            minutes: Time window for calculations
        
        Returns:
            Dict with statistics
        """
        volume = self.get_volume_for_interval(minutes)
        flow_rate = self.get_flow_rate(minutes)
        flow_rate_hourly = self.get_flow_rate_per_hour(minutes)
        
        return {
            'volume': volume,
            'flow_rate_per_min': flow_rate,
            'flow_rate_per_hour': flow_rate_hourly,
            'vehicles': self.get_vehicle_count(minutes),
            'pedestrians': self.get_pedestrian_count(minutes),
            'by_class': self.get_class_distribution(minutes),
            'by_direction': self.get_direction_distribution(minutes),
            'time_window_minutes': minutes
        }
    
    def get_all_statistics(self) -> Dict:
        """Get statistics for all configured intervals"""
        return {
            'session_total': self.get_total_volume(),
            'session_duration_seconds': self.get_session_duration().total_seconds(),
            'average_flow_per_minute': self.get_average_flow_rate(),
            'class_distribution': self.get_class_distribution(),
            'direction_distribution': self.get_direction_distribution(),
            'by_interval': {
                minutes: self.get_statistics(minutes)
                for minutes in self.intervals
            }
        }
    
    def get_time_series_data(self, minutes: int = 5, 
                            history_minutes: int = 60) -> List[Tuple[datetime, int]]:
        """
        Get volume over time
        
        Args:
            minutes: Interval size
            history_minutes: How far back to look
        
        Returns:
            List of (timestamp, volume) tuples
        """
        cutoff_time = time.time() - (history_minutes * 60)
        interval_seconds = minutes * 60
        
        # Group events into intervals
        intervals_dict: Dict[int, int] = defaultdict(int)
        
        for event in self.events:
            event_time = event.get('timestamp', 0)
            if event_time > cutoff_time:
                # Round down to interval boundary
                interval_bucket = int(event_time / interval_seconds) * interval_seconds
                intervals_dict[interval_bucket] += 1
        
        # Create time series
        result = []
        for interval_bucket in sorted(intervals_dict.keys()):
            ts = datetime.fromtimestamp(interval_bucket)
            count = intervals_dict[interval_bucket]
            result.append((ts, count))
        
        return result
    
    def get_peak_interval(self, minutes: int = 5,
                         history_minutes: int = 60) -> Tuple[Optional[datetime], int]:
        """
        Get peak volume interval
        
        Args:
            minutes: Interval size
            history_minutes: How far back to look
        
        Returns:
            Tuple of (peak_time, peak_volume)
        """
        time_series = self.get_time_series_data(minutes, history_minutes)
        
        if not time_series:
            return None, 0
        
        peak_time, peak_volume = max(time_series, key=lambda x: x[1])
        return peak_time, peak_volume
    
    def reset(self):
        """Reset analytics"""
        self.events.clear()
        self.class_counts.clear()
        self.direction_counts.clear()
        self.session_start_time = datetime.now()
        self.peak_time = None
        self.peak_volume = 0
    
    def get_percentile_by_class(self) -> Dict[str, float]:
        """Get percentage distribution by class"""
        total = len(self.events)
        if total == 0:
            return {}
        
        return {
            cls: (count / total) * 100
            for cls, count in self.class_counts.items()
        }
    
    def get_percentile_by_direction(self) -> Dict[str, float]:
        """Get percentage distribution by direction"""
        total = len(self.events)
        if total == 0:
            return {}
        
        return {
            direction: (count / total) * 100
            for direction, count in self.direction_counts.items()
        }
    
    def clear_old_events(self, hours: int = 24):
        """Clear events older than specified hours"""
        cutoff_time = time.time() - (hours * 3600)
        
        # Remove old events from main deque
        while self.events and self.events[0].get('timestamp', 0) < cutoff_time:
            self.events.popleft()
