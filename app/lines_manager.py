"""
Professional counting lines management system
Supports multiple lines with bidirectional counting and traffic engineering calculations
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import threading
import json
from pathlib import Path


@dataclass
class CountingEvent:
    """A crossing event through a line"""
    timestamp: datetime
    track_id: int
    class_id: int
    class_label: str
    confidence: float
    direction: str  # "A->B" or "B->A"
    line_id: int
    position_x: float
    position_y: float


@dataclass
class CountingLine:
    """Professional counting line with name, directions, and thresholds"""
    id: int
    name: str
    norm_x1: float  # Normalized [0, 1]
    norm_y1: float
    norm_x2: float
    norm_y2: float
    direction_a: str  # "Norte", "Sur", etc.
    direction_b: str
    enabled: bool = True
    frame_height: int = 480
    frame_width: int = 640
    
    # For crossing detection
    crossing_history: Dict[Tuple[int, str], datetime] = field(default_factory=dict)  # (track_id, direction) -> last_crossing_time
    
    def get_pixel_coords(self) -> Tuple[int, int, int, int]:
        """Convert normalized to pixel coordinates"""
        x1 = int(self.norm_x1 * self.frame_width)
        y1 = int(self.norm_y1 * self.frame_height)
        x2 = int(self.norm_x2 * self.frame_width)
        y2 = int(self.norm_y2 * self.frame_height)
        return x1, y1, x2, y2
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'norm_x1': self.norm_x1,
            'norm_y1': self.norm_y1,
            'norm_x2': self.norm_x2,
            'norm_y2': self.norm_y2,
            'direction_a': self.direction_a,
            'direction_b': self.direction_b,
            'enabled': self.enabled,
        }


@dataclass
class TrafficFlowStats:
    """Traffic flow statistics for an interval"""
    interval_minutes: int
    start_time: datetime
    end_time: datetime
    
    total_count: int = 0
    vehicles_count: int = 0
    pedestrians_count: int = 0
    
    direction_ab: int = 0
    direction_ba: int = 0
    
    # By class
    cars: int = 0
    motorcycles: int = 0
    buses: int = 0
    trucks: int = 0
    bicycles: int = 0
    persons: int = 0
    
    # Flow rates
    flow_rate_per_min: float = 0.0  # veh/min
    flow_rate_per_hour: float = 0.0  # veh/h
    
    # Peak hour factor (only if interval is 15 min and hour complete)
    phf: Optional[float] = None
    
    def calculate_flow_rates(self):
        """Calculate flow rates from counts"""
        elapsed_minutes = (self.end_time - self.start_time).total_seconds() / 60.0
        if elapsed_minutes > 0:
            self.flow_rate_per_min = self.total_count / elapsed_minutes
            self.flow_rate_per_hour = self.flow_rate_per_min * 60
    
    def get_class_distribution(self) -> Dict[str, float]:
        """Get percentage distribution by class"""
        if self.total_count == 0:
            return {}
        
        return {
            'Carros': (self.cars / self.total_count) * 100,
            'Motos': (self.motorcycles / self.total_count) * 100,
            'Buses': (self.buses / self.total_count) * 100,
            'Camiones': (self.trucks / self.total_count) * 100,
            'Bicicletas': (self.bicycles / self.total_count) * 100,
            'Personas': (self.persons / self.total_count) * 100,
        }


class LinesManager:
    """Manage multiple counting lines with traffic engineering calculations"""
    
    VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck
    PEDESTRIAN_CLASSES = {0}  # person
    BICYCLE_CLASSES = {1}
    
    CLASS_MAPPING = {
        0: ('Persona', False),
        1: ('Bicicleta', True),
        2: ('Carro', True),
        3: ('Moto', True),
        5: ('Bus', True),
        7: ('Camión', True),
    }
    
    def __init__(self, config_path: str = "config/lines.json"):
        self.lines: Dict[int, CountingLine] = {}
        self.events: List[CountingEvent] = []
        self.config_path = Path(config_path)
        self.lock = threading.Lock()
        self.next_line_id = 1
        
        # Statistics by line and interval
        self.stats: Dict[int, Dict[int, TrafficFlowStats]] = {}  # line_id -> {interval_minutes -> stats}
        
        # Load from config if exists
        self._load_config()
    
    def add_line(self, name: str, norm_x1: float, norm_y1: float, 
                 norm_x2: float, norm_y2: float,
                 direction_a: str = "A", direction_b: str = "B",
                 frame_height: int = 480, frame_width: int = 640) -> CountingLine:
        """Add a new counting line"""
        with self.lock:
            line = CountingLine(
                id=self.next_line_id,
                name=name,
                norm_x1=norm_x1,
                norm_y1=norm_y1,
                norm_x2=norm_x2,
                norm_y2=norm_y2,
                direction_a=direction_a,
                direction_b=direction_b,
                frame_height=frame_height,
                frame_width=frame_width,
            )
            
            self.lines[self.next_line_id] = line
            self.stats[self.next_line_id] = {}
            self.next_line_id += 1
            
            self._save_config()
            return line
    
    def remove_line(self, line_id: int) -> bool:
        """Remove a counting line"""
        with self.lock:
            if line_id in self.lines:
                del self.lines[line_id]
                if line_id in self.stats:
                    del self.stats[line_id]
                self._save_config()
                return True
        return False
    
    def update_line(self, line_id: int, **kwargs) -> bool:
        """Update line properties"""
        with self.lock:
            if line_id not in self.lines:
                return False
            
            line = self.lines[line_id]
            for key, value in kwargs.items():
                if hasattr(line, key):
                    setattr(line, key, value)
            
            self._save_config()
            return True
    
    def get_lines(self) -> Dict[int, CountingLine]:
        """Get all lines"""
        with self.lock:
            return dict(self.lines)
    
    def get_line(self, line_id: int) -> Optional[CountingLine]:
        """Get a specific line"""
        with self.lock:
            return self.lines.get(line_id)
    
    def record_crossing(self, event: CountingEvent) -> bool:
        """Record a crossing event"""
        with self.lock:
            # Check if this is a duplicate crossing (same track_id, direction within 0.5 seconds)
            key = (event.track_id, event.direction)
            if key in self.lines[event.line_id].crossing_history:
                last_time = self.lines[event.line_id].crossing_history[key]
                if (event.timestamp - last_time).total_seconds() < 0.5:
                    return False  # Duplicate crossing
            
            # Record crossing
            self.lines[event.line_id].crossing_history[key] = event.timestamp
            self.events.append(event)
            
            # Update statistics
            self._update_stats(event)
            
            return True
    
    def _update_stats(self, event: CountingEvent):
        """Update traffic statistics"""
        line_id = event.line_id
        
        # Ensure intervals exist
        for minutes in [1, 5, 15, 60]:
            if minutes not in self.stats[line_id]:
                self.stats[line_id][minutes] = TrafficFlowStats(
                    interval_minutes=minutes,
                    start_time=event.timestamp,
                    end_time=event.timestamp + timedelta(minutes=minutes)
                )
        
        # Update all relevant intervals
        for minutes, stats in self.stats[line_id].items():
            if stats.start_time <= event.timestamp <= stats.end_time:
                stats.total_count += 1
                
                # Update direction
                if event.direction == "A->B":
                    stats.direction_ab += 1
                else:
                    stats.direction_ba += 1
                
                # Update class
                if event.class_id in self.VEHICLE_CLASSES:
                    stats.vehicles_count += 1
                    if event.class_id == 2:
                        stats.cars += 1
                    elif event.class_id == 3:
                        stats.motorcycles += 1
                    elif event.class_id == 5:
                        stats.buses += 1
                    elif event.class_id == 7:
                        stats.trucks += 1
                
                elif event.class_id in self.PEDESTRIAN_CLASSES:
                    stats.pedestrians_count += 1
                    stats.persons += 1
                
                elif event.class_id in self.BICYCLE_CLASSES:
                    stats.bicycles += 1
                
                stats.calculate_flow_rates()
    
    def get_volume(self, line_id: int, interval_minutes: int = 1) -> int:
        """Get vehicle count for interval"""
        with self.lock:
            if line_id not in self.stats or interval_minutes not in self.stats[line_id]:
                return 0
            return self.stats[line_id][interval_minutes].total_count
    
    def get_flow_rate(self, line_id: int, interval_minutes: int = 1, 
                      unit: str = "veh/h") -> float:
        """Get flow rate in specified unit"""
        with self.lock:
            if line_id not in self.stats or interval_minutes not in self.stats[line_id]:
                return 0.0
            
            stats = self.stats[line_id][interval_minutes]
            if unit == "veh/h":
                return stats.flow_rate_per_hour
            elif unit == "veh/min":
                return stats.flow_rate_per_min
            return 0.0
    
    def get_stats(self, line_id: int, interval_minutes: int = 1) -> Optional[TrafficFlowStats]:
        """Get complete statistics for interval"""
        with self.lock:
            if line_id not in self.stats or interval_minutes not in self.stats[line_id]:
                return None
            return self.stats[line_id][interval_minutes]
    
    def get_all_stats(self, line_id: int) -> Dict[int, TrafficFlowStats]:
        """Get all interval statistics"""
        with self.lock:
            return dict(self.stats.get(line_id, {}))
    
    def get_events(self, line_id: int = None, last_n: int = 50) -> List[CountingEvent]:
        """Get recent crossing events"""
        with self.lock:
            events = self.events if line_id is None else [e for e in self.events if e.line_id == line_id]
            return events[-last_n:]
    
    def calculate_phf(self, line_id: int) -> Optional[float]:
        """Calculate Peak Hour Factor if sufficient data"""
        with self.lock:
            if line_id not in self.stats or 15 not in self.stats[line_id]:
                return None
            
            # Need at least 4 complete 15-minute intervals (1 hour)
            fifteen_min_intervals = self.stats[line_id][15]
            
            # This is simplified - in production would track hourly data
            # For now, return None until we have proper hourly aggregation
            return None
    
    def _save_config(self):
        """Save lines configuration to file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                'lines': {str(line_id): line.to_dict() for line_id, line in self.lines.items()},
                'next_line_id': self.next_line_id,
            }
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving lines config: {e}")
    
    def _load_config(self):
        """Load lines configuration from file"""
        try:
            if not self.config_path.exists():
                # Create default line
                self.add_line(
                    name="Línea Principal",
                    norm_x1=0.0,
                    norm_y1=0.5,
                    norm_x2=1.0,
                    norm_y2=0.5,
                    direction_a="Izquierda",
                    direction_b="Derecha"
                )
                return
            
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            self.next_line_id = config.get('next_line_id', 1)
            
            for line_id_str, line_data in config.get('lines', {}).items():
                line_id = int(line_id_str)
                line = CountingLine(
                    id=line_id,
                    name=line_data['name'],
                    norm_x1=line_data['norm_x1'],
                    norm_y1=line_data['norm_y1'],
                    norm_x2=line_data['norm_x2'],
                    norm_y2=line_data['norm_y2'],
                    direction_a=line_data.get('direction_a', 'A'),
                    direction_b=line_data.get('direction_b', 'B'),
                    enabled=line_data.get('enabled', True),
                )
                self.lines[line_id] = line
                self.stats[line_id] = {}
        
        except Exception as e:
            print(f"Error loading lines config: {e}")
            # Create default line if load fails
            if not self.lines:
                self.add_line(
                    name="Línea Principal",
                    norm_x1=0.0,
                    norm_y1=0.5,
                    norm_x2=1.0,
                    norm_y2=0.5,
                    direction_a="Izquierda",
                    direction_b="Derecha"
                )
    
    def reset(self):
        """Reset all statistics"""
        with self.lock:
            self.events.clear()
            for line_id in self.lines:
                self.stats[line_id].clear()
                self.lines[line_id].crossing_history.clear()
