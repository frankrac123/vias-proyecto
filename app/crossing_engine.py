"""Track-based, stateful crossing engine for Traffic Vision V3."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from math import hypot
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Tuple
import time
import uuid

Point = Tuple[float, float]
SIDE_A, SIDE_B, ON_LINE = "SIDE_A", "SIDE_B", "ON_LINE"


def cross(a: Point, b: Point, p: Point) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def segment_intersection(a: Point, b: Point, c: Point, d: Point,
                         eps: float = 1e-9) -> Optional[Point]:
    """Return the finite-segment intersection, including endpoint contact."""
    rx, ry = b[0] - a[0], b[1] - a[1]
    sx, sy = d[0] - c[0], d[1] - c[1]
    den = rx * sy - ry * sx
    qpx, qpy = c[0] - a[0], c[1] - a[1]
    if abs(den) <= eps:
        return None
    t = (qpx * sy - qpy * sx) / den
    u = (qpx * ry - qpy * rx) / den
    if -eps <= t <= 1 + eps and -eps <= u <= 1 + eps:
        return a[0] + t * rx, a[1] + t * ry
    return None


def point_in_polygon(point: Point, polygon: Optional[List[Point]]) -> bool:
    if not polygon or len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


@dataclass
class CrossingLine:
    line_id: int
    name: str
    p1: Point
    p2: Point
    side_a_name: str = "A"
    side_b_name: str = "B"
    enabled: bool = True
    mode: str = "basic"
    zone_a: Optional[List[Point]] = None
    zone_b: Optional[List[Point]] = None


@dataclass
class CrossingEvent:
    event_id: str
    timestamp: str
    session_id: int
    camera_id: str
    logical_track_id: int
    tracker_id: int
    class_id: int
    class_name: str
    confidence: float
    line_id: int
    line_name: str
    side_from: str
    side_to: str
    direction_name: str
    x: float
    y: float
    speed_px_s: float
    speed_kmh: Optional[float] = None
    lane_id: Optional[str] = None
    quality_score: float = 0.0
    frame_id: int = 0

    def to_dict(self) -> Dict:
        return dict(self.__dict__)

    @property
    def track_id(self):
        return self.tracker_id

    @property
    def class_label(self):
        return self.class_name

    @property
    def direction(self):
        return "A->B" if self.side_from == SIDE_A else "B->A"

    @property
    def position_x(self):
        return self.x

    @property
    def position_y(self):
        return self.y


@dataclass
class LineState:
    stable_side: Optional[str] = None
    previous_side: Optional[str] = None
    current_side: Optional[str] = None
    pending_from: Optional[str] = None
    pending_to: Optional[str] = None
    confirmation_count: int = 0
    intersection: Optional[Point] = None
    crossing_state: str = "UNSEEN"
    last_event_time: float = -1e30
    saw_zone_a: bool = False
    saw_zone_b: bool = False
    sequence: int = 0


@dataclass
class LogicalTrack:
    logical_track_id: int
    tracker_id: int
    first_seen: float
    last_seen: float
    lost_frames: int = 0
    raw_trajectory: Deque[Point] = field(default_factory=lambda: deque(maxlen=60))
    trajectory: Deque[Point] = field(default_factory=lambda: deque(maxlen=60))
    class_votes: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    class_names: Dict[int, str] = field(default_factory=dict)
    confidences: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    line_states: Dict[int, LineState] = field(default_factory=dict)
    bbox: Optional[Tuple[float, float, float, float]] = None

    @property
    def class_id(self) -> int:
        return max(self.class_votes, key=self.class_votes.get)

    @property
    def class_name(self) -> str:
        return self.class_names.get(self.class_id, str(self.class_id))

    @property
    def confidence(self) -> float:
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0


class CrossingEngine:
    def __init__(self, session_id: int = 0, camera_id: str = "camera-1",
                 confirmation_frames: int = 2, dead_band_px: float = 8.0,
                 line_reentry_seconds: float = 2.0, max_lost_frames: int = 30,
                 id_switch_distance_px: float = 80.0, ema_alpha: float = 0.65,
                 class_thresholds: Optional[Mapping[str, float]] = None):
        self.session_id = session_id
        self.camera_id = camera_id
        self.confirmation_frames = max(1, confirmation_frames)
        self.dead_band_px = max(0.0, dead_band_px)
        self.line_reentry_seconds = max(0.0, line_reentry_seconds)
        self.max_lost_frames = max_lost_frames
        self.id_switch_distance_px = id_switch_distance_px
        self.ema_alpha = ema_alpha
        self.class_thresholds = dict(class_thresholds or {})
        self.lines: Dict[int, CrossingLine] = {}
        self.tracks: Dict[int, LogicalTrack] = {}
        self.tracker_to_logical: Dict[int, int] = {}
        self.next_logical_id = 1
        self.events: List[CrossingEvent] = []

    def set_lines(self, lines: Iterable[CrossingLine]) -> None:
        self.lines = {line.line_id: line for line in lines}

    def get_events(self, line_id: Optional[int] = None, last_n: int = 100):
        events = self.events[-last_n:]
        return [e for e in events if e.line_id == line_id] if line_id is not None else events

    def get_line(self, line_id: int):
        return self.lines.get(line_id)

    def get_count(self, line_id: int, direction: Optional[str] = None) -> int:
        return sum(1 for event in self.events if event.line_id == line_id and
                   (direction is None or event.direction == direction))

    def get_count_by_class(self, line_id: int, direction: Optional[str] = None) -> Dict[str, int]:
        counts = Counter(event.class_name for event in self.events if event.line_id == line_id and
                         (direction is None or event.direction == direction))
        return dict(counts)

    def clear_history(self):
        self.events.clear()

    def _side(self, point: Point, line: CrossingLine) -> str:
        length = max(hypot(line.p2[0] - line.p1[0], line.p2[1] - line.p1[1]), 1e-9)
        distance = cross(line.p1, line.p2, point) / length
        if abs(distance) <= self.dead_band_px:
            return ON_LINE
        return SIDE_A if distance > 0 else SIDE_B

    @staticmethod
    def _track_data(track) -> Tuple[int, str, float, Point, Optional[tuple]]:
        pos = track.get_last_position()
        return (track.class_id, getattr(track, "class_name", str(track.class_id)),
                float(track.confidence), pos, getattr(track, "bbox", None))

    def _logical_for(self, tracker_id: int, class_id: int, position: Point,
                     now: float) -> LogicalTrack:
        logical_id = self.tracker_to_logical.get(tracker_id)
        if logical_id in self.tracks:
            return self.tracks[logical_id]
        candidate = None
        candidate_distance = self.id_switch_distance_px
        for logical in self.tracks.values():
            if logical.lost_frames <= 0 or logical.lost_frames > self.max_lost_frames:
                continue
            if logical.class_votes and logical.class_id != class_id:
                continue
            if not logical.trajectory:
                continue
            distance = hypot(position[0] - logical.trajectory[-1][0], position[1] - logical.trajectory[-1][1])
            if distance < candidate_distance:
                candidate, candidate_distance = logical, distance
        if candidate is None:
            candidate = LogicalTrack(self.next_logical_id, tracker_id, now, now)
            self.tracks[candidate.logical_track_id] = candidate
            self.next_logical_id += 1
        candidate.tracker_id = tracker_id
        candidate.lost_frames = 0
        self.tracker_to_logical[tracker_id] = candidate.logical_track_id
        return candidate

    def process_tracks(self, tracks: Mapping[int, object], frame_id: int,
                       timestamp: Optional[float] = None, camera_fps: float = 0.0) -> List[CrossingEvent]:
        now = time.monotonic() if timestamp is None else float(timestamp)
        seen_logical = set()
        new_events: List[CrossingEvent] = []
        for tracker_id, source in tracks.items():
            class_id, class_name, confidence, position, bbox = self._track_data(source)
            if position is None or confidence < self.class_thresholds.get(class_name, 0.0):
                continue
            logical = self._logical_for(tracker_id, class_id, position, now)
            seen_logical.add(logical.logical_track_id)
            logical.last_seen, logical.lost_frames, logical.bbox = now, 0, bbox
            logical.class_votes[class_id] += max(confidence, 0.01)
            logical.class_names[class_id] = class_name
            logical.confidences.append(confidence)
            logical.raw_trajectory.append(position)
            if logical.trajectory:
                old = logical.trajectory[-1]
                filtered = (self.ema_alpha * position[0] + (1-self.ema_alpha) * old[0],
                            self.ema_alpha * position[1] + (1-self.ema_alpha) * old[1])
            else:
                filtered = position
            previous = logical.trajectory[-1] if logical.trajectory else None
            logical.trajectory.append(filtered)
            for line in self.lines.values():
                if line.enabled:
                    event = self._process_line(logical, line, previous, filtered, now, frame_id, camera_fps)
                    if event:
                        self.events.append(event)
                        new_events.append(event)
        for logical_id, logical in list(self.tracks.items()):
            if logical_id not in seen_logical:
                logical.lost_frames += 1
                if logical.lost_frames > self.max_lost_frames:
                    self.tracks.pop(logical_id)
        return new_events

    def _process_line(self, track: LogicalTrack, line: CrossingLine, previous: Optional[Point],
                      current: Point, now: float, frame_id: int, camera_fps: float) -> Optional[CrossingEvent]:
        state = track.line_states.setdefault(line.line_id, LineState())
        side = self._side(current, line)
        state.previous_side, state.current_side = state.current_side, side
        state.saw_zone_a |= point_in_polygon(current, line.zone_a)
        state.saw_zone_b |= point_in_polygon(current, line.zone_b)
        if side != ON_LINE and state.stable_side is None:
            state.stable_side = side
            state.crossing_state = "APPROACHING_A" if side == SIDE_A else "APPROACHING_B"
            return None
        if state.pending_to:
            if side == state.pending_to:
                state.confirmation_count += 1
            elif side not in (ON_LINE, state.pending_to):
                state.pending_from = state.pending_to = None
                state.confirmation_count = 0
        elif previous is not None and side != ON_LINE and state.stable_side not in (None, side):
            hit = segment_intersection(previous, current, line.p1, line.p2)
            zone_ok = line.mode != "robust" or (
                state.saw_zone_a if state.stable_side == SIDE_A else state.saw_zone_b
            )
            if hit and zone_ok:
                state.pending_from, state.pending_to = state.stable_side, side
                state.intersection, state.confirmation_count = hit, 1
                state.crossing_state = "CROSSING"
        if state.pending_to and state.confirmation_count >= self.confirmation_frames:
            robust_destination = line.mode != "robust" or (
                point_in_polygon(current, line.zone_b) if state.pending_to == SIDE_B
                else point_in_polygon(current, line.zone_a)
            )
            if robust_destination and now - state.last_event_time > self.line_reentry_seconds:
                event = self._make_event(track, line, state, now, frame_id, camera_fps)
                state.last_event_time, state.sequence = now, state.sequence + 1
                state.crossing_state = "CONFIRMED_A_TO_B" if state.pending_from == SIDE_A else "CONFIRMED_B_TO_A"
                state.stable_side = state.pending_to
                state.pending_from = state.pending_to = None
                state.confirmation_count = 0
                return event
        return None

    def _make_event(self, track: LogicalTrack, line: CrossingLine, state: LineState,
                    now: float, frame_id: int, camera_fps: float) -> CrossingEvent:
        p = state.intersection or track.trajectory[-1]
        speed = 0.0
        if len(track.trajectory) >= 2:
            speed = hypot(track.trajectory[-1][0]-track.trajectory[-2][0],
                          track.trajectory[-1][1]-track.trajectory[-2][1]) * max(camera_fps, 1.0)
        direction = (f"{line.side_a_name} -> {line.side_b_name}" if state.pending_from == SIDE_A
                     else f"{line.side_b_name} -> {line.side_a_name}")
        stability = track.class_votes[track.class_id] / max(sum(track.class_votes.values()), 1e-9)
        quality = min(1.0, .45*track.confidence + .25*stability + .2*min(len(track.trajectory)/10, 1) + .1*(1-min(track.lost_frames/5, 1)))
        return CrossingEvent(str(uuid.uuid4()), datetime.now().isoformat(), self.session_id,
                             self.camera_id, track.logical_track_id, track.tracker_id,
                             track.class_id, track.class_name, track.confidence,
                             line.line_id, line.name, state.pending_from, state.pending_to,
                             direction, p[0], p[1], speed, quality_score=quality,
                             frame_id=frame_id)
