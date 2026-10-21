"""
SQLite Database module for persistent event storage
"""
import sqlite3
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict
import json


class DatabaseManager:
    """Thread-safe SQLite database manager"""
    
    def __init__(self, db_path: str = "./data/traffic.db"):
        """Initialize database"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        
        self._initialize_db()
    
    def _get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _initialize_db(self):
        """Create tables if they don't exist"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration_seconds INTEGER,
                    camera_source TEXT,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Crossing events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS crossing_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    frame_id INTEGER,
                    track_id INTEGER NOT NULL,
                    class_id INTEGER,
                    class_label TEXT,
                    line_id INTEGER,
                    line_name TEXT,
                    direction TEXT,
                    position_x REAL,
                    position_y REAL,
                    confidence REAL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            ''')
            
            # Analytics intervals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analytics_intervals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    line_id INTEGER,
                    interval_minutes INTEGER,
                    interval_start TEXT,
                    interval_end TEXT,
                    vehicle_count INTEGER DEFAULT 0,
                    pedestrian_count INTEGER DEFAULT 0,
                    direction_ab INTEGER DEFAULT 0,
                    direction_ba INTEGER DEFAULT 0,
                    count_by_class TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            ''')
            
            # Create indexes for better query performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_events ON crossing_events(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_line ON crossing_events(line_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_timestamp ON crossing_events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_track ON crossing_events(track_id)')

            # Non-destructive V3 migration for richer confirmed crossing events.
            existing = {row[1] for row in cursor.execute('PRAGMA table_info(crossing_events)')}
            v3_columns = {
                'event_id': 'TEXT', 'camera_id': 'TEXT', 'logical_track_id': 'INTEGER',
                'tracker_id': 'INTEGER', 'class_name': 'TEXT', 'side_from': 'TEXT',
                'side_to': 'TEXT', 'direction_name': 'TEXT', 'speed_px_s': 'REAL',
                'speed_kmh': 'REAL', 'lane_id': 'TEXT', 'quality_score': 'REAL'
            }
            for name, sql_type in v3_columns.items():
                if name not in existing:
                    cursor.execute(f'ALTER TABLE crossing_events ADD COLUMN {name} {sql_type}')
            
            conn.commit()
            conn.close()
    
    def create_session(self, camera_source: str = "", notes: str = "") -> int:
        """
        Create new session
        
        Returns:
            Session ID
        """
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sessions (start_time, camera_source, notes)
                VALUES (?, ?, ?)
            ''', (datetime.now().isoformat(), camera_source, notes))
            
            session_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return session_id
    
    def end_session(self, session_id: int):
        """End session and calculate duration"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE sessions
                SET end_time = ?,
                    duration_seconds = (
                        SELECT CAST((julianday(?) - julianday(start_time)) * 86400 AS INTEGER)
                    )
                WHERE id = ?
            ''', (datetime.now().isoformat(), datetime.now().isoformat(), session_id))
            
            conn.commit()
            conn.close()
    
    def add_crossing_event(self, session_id: int, crossing_event: Dict):
        """Add crossing event to database"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO crossing_events
                (session_id, timestamp, frame_id, track_id, class_id, class_label,
                 line_id, line_name, direction, position_x, position_y, confidence,
                 event_id, camera_id, logical_track_id, tracker_id, class_name,
                 side_from, side_to, direction_name, speed_px_s, speed_kmh, lane_id, quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                crossing_event.get('timestamp', datetime.now().isoformat()),
                crossing_event.get('frame_id'),
                crossing_event.get('track_id', crossing_event.get('tracker_id')),
                crossing_event.get('class_id'),
                crossing_event.get('class_label', crossing_event.get('class_name')),
                crossing_event.get('line_id'),
                crossing_event.get('line_name', 'Unknown'),
                crossing_event.get('direction', crossing_event.get('direction_name')),
                crossing_event.get('position_x', crossing_event.get('x')),
                crossing_event.get('position_y', crossing_event.get('y')),
                crossing_event.get('confidence'), crossing_event.get('event_id'),
                crossing_event.get('camera_id'), crossing_event.get('logical_track_id'),
                crossing_event.get('tracker_id'), crossing_event.get('class_name'),
                crossing_event.get('side_from'), crossing_event.get('side_to'),
                crossing_event.get('direction_name'), crossing_event.get('speed_px_s'),
                crossing_event.get('speed_kmh'), crossing_event.get('lane_id'),
                crossing_event.get('quality_score')
            ))
            
            conn.commit()
            conn.close()
    
    def add_analytics_interval(self, session_id: int, interval_data: Dict):
        """Add analytics interval record"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO analytics_intervals
                (session_id, line_id, interval_minutes, interval_start, interval_end,
                 vehicle_count, pedestrian_count, direction_ab, direction_ba, count_by_class)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                interval_data.get('line_id'),
                interval_data.get('interval_minutes'),
                interval_data.get('interval_start'),
                interval_data.get('interval_end'),
                interval_data.get('vehicle_count', 0),
                interval_data.get('pedestrian_count', 0),
                interval_data.get('direction_ab', 0),
                interval_data.get('direction_ba', 0),
                json.dumps(interval_data.get('count_by_class', {}))
            ))
            
            conn.commit()
            conn.close()
    
    def get_session_events(self, session_id: int, line_id: Optional[int] = None) -> List[Dict]:
        """Get all events for a session"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            if line_id is None:
                cursor.execute('''
                    SELECT * FROM crossing_events
                    WHERE session_id = ?
                    ORDER BY timestamp
                ''', (session_id,))
            else:
                cursor.execute('''
                    SELECT * FROM crossing_events
                    WHERE session_id = ? AND line_id = ?
                    ORDER BY timestamp
                ''', (session_id, line_id))
            
            events = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return events
    
    def get_session_analytics(self, session_id: int) -> List[Dict]:
        """Get analytics for session"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM analytics_intervals
                WHERE session_id = ?
                ORDER BY interval_start
            ''', (session_id,))
            
            intervals = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return intervals
    
    def get_session_info(self, session_id: int) -> Optional[Dict]:
        """Get session information"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM sessions WHERE id = ?', (session_id,))
            row = cursor.fetchone()
            conn.close()
            
            return dict(row) if row else None
    
    def get_count_summary(self, session_id: int, line_id: Optional[int] = None) -> Dict:
        """Get count summary for session"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            if line_id is None:
                # Total counts by class
                cursor.execute('''
                    SELECT class_label, COUNT(*) as count
                    FROM crossing_events
                    WHERE session_id = ?
                    GROUP BY class_label
                ''', (session_id,))
            else:
                cursor.execute('''
                    SELECT class_label, COUNT(*) as count
                    FROM crossing_events
                    WHERE session_id = ? AND line_id = ?
                    GROUP BY class_label
                ''', (session_id, line_id))
            
            summary = {row['class_label']: row['count'] for row in cursor.fetchall()}
            conn.close()
            
            return summary
    
    def get_direction_summary(self, session_id: int, line_id: Optional[int] = None) -> Dict:
        """Get count by direction"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            if line_id is None:
                cursor.execute('''
                    SELECT direction, COUNT(*) as count
                    FROM crossing_events
                    WHERE session_id = ?
                    GROUP BY direction
                ''', (session_id,))
            else:
                cursor.execute('''
                    SELECT direction, COUNT(*) as count
                    FROM crossing_events
                    WHERE session_id = ? AND line_id = ?
                    GROUP BY direction
                ''', (session_id, line_id))
            
            summary = {row['direction']: row['count'] for row in cursor.fetchall()}
            conn.close()
            
            return summary
    
    def get_event_time_series(self, session_id: int, line_id: Optional[int] = None,
                             interval_minutes: int = 5) -> List[Tuple[str, int]]:
        """
        Get event counts over time intervals
        
        Returns:
            List of (timestamp, count) tuples
        """
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            if line_id is None:
                cursor.execute(f'''
                    SELECT 
                        datetime(timestamp, 'start of {interval_minutes} minutes') as interval,
                        COUNT(*) as count
                    FROM crossing_events
                    WHERE session_id = ?
                    GROUP BY interval
                    ORDER BY interval
                ''', (session_id,))
            else:
                cursor.execute(f'''
                    SELECT 
                        datetime(timestamp, 'start of {interval_minutes} minutes') as interval,
                        COUNT(*) as count
                    FROM crossing_events
                    WHERE session_id = ? AND line_id = ?
                    GROUP BY interval
                    ORDER BY interval
                ''', (session_id, line_id))
            
            series = [(row['interval'], row['count']) for row in cursor.fetchall()]
            conn.close()
            
            return series
    
    def delete_session(self, session_id: int):
        """Delete session and all related data"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM crossing_events WHERE session_id = ?', (session_id,))
            cursor.execute('DELETE FROM analytics_intervals WHERE session_id = ?', (session_id,))
            cursor.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
            
            conn.commit()
            conn.close()
    
    def export_session_to_dict(self, session_id: int) -> Dict:
        """Export session data as dictionary"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get session info
            cursor.execute('SELECT * FROM sessions WHERE id = ?', (session_id,))
            session = dict(cursor.fetchone()) if cursor.fetchone() else {}
            
            # Get events
            cursor.execute('SELECT * FROM crossing_events WHERE session_id = ?', (session_id,))
            events = [dict(row) for row in cursor.fetchall()]
            
            # Get analytics
            cursor.execute('SELECT * FROM analytics_intervals WHERE session_id = ?', (session_id,))
            analytics = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            
            return {
                'session': session,
                'events': events,
                'analytics': analytics
            }
