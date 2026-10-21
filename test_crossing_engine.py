import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from app.crossing_engine import CrossingEngine, CrossingLine, segment_intersection
from app.database import DatabaseManager
from app.analytics import AnalyticsEngine


@dataclass
class FakeTrack:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    pos: tuple
    bbox: tuple = (0, 0, 10, 10)

    def get_last_position(self):
        return self.pos


class CrossingEngineTests(unittest.TestCase):
    def setUp(self):
        self.line = CrossingLine(1, "Principal", (0, -10), (0, 10), "OESTE", "ESTE")
        self.engine = CrossingEngine(7, confirmation_frames=2, dead_band_px=0,
                                     line_reentry_seconds=2, ema_alpha=1)
        self.engine.set_lines([self.line])

    def feed(self, track_id, positions, start=0, class_id=2, name="car"):
        events = []
        for i, pos in enumerate(positions):
            track = FakeTrack(track_id, class_id, name, .9, pos)
            events += self.engine.process_tracks({track_id: track}, i, start+i, 30)
        return events

    def test_a_to_b_normal(self):
        events = self.feed(1, [(-5, 0), (-2, 0), (2, 0), (4, 0)])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].direction_name, "OESTE -> ESTE")

    def test_b_to_a_normal(self):
        events = self.feed(1, [(5, 0), (2, 0), (-2, 0), (-4, 0)])
        self.assertEqual(events[0].direction_name, "ESTE -> OESTE")

    def test_fast_vehicle(self):
        line = CrossingLine(1, "x300", (300, 0), (300, 200))
        self.engine.set_lines([line])
        events = self.feed(1, [(100, 100), (500, 100), (550, 100)])
        self.assertEqual(len(events), 1)

    def test_parallel_without_crossing(self):
        self.assertEqual(self.feed(1, [(-5, -5), (-5, 0), (-5, 5)]), [])

    def test_outside_line_endpoint(self):
        self.assertEqual(self.feed(1, [(-5, 30), (5, 30), (7, 30)]), [])

    def test_on_line_without_crossing(self):
        self.assertEqual(self.feed(1, [(0, -5), (0, 0), (0, 5)]), [])

    def test_vibration_is_not_double_counted(self):
        self.engine.dead_band_px = 2
        events = self.feed(1, [(-5, 0), (1, 0), (-1, 0), (1, 0), (5, 0), (6, 0)])
        self.assertLessEqual(len(events), 1)

    def test_double_counting_cooldown(self):
        events = self.feed(1, [(-5, 0), (5, 0), (6, 0), (-5, 0), (-6, 0)], start=0)
        self.assertEqual(len(events), 1)

    def test_valid_reentry(self):
        first = self.feed(1, [(-5, 0), (5, 0), (6, 0)], start=0)
        second = self.feed(1, [(-5, 0), (-6, 0)], start=10)
        self.assertEqual(len(first + second), 2)

    def test_temporary_track_loss_and_id_switch(self):
        self.feed(10, [(-5, 0)], start=0)
        self.engine.process_tracks({}, 2, 1, 30)
        events = self.feed(11, [(5, 0), (6, 0)], start=2)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].logical_track_id, 1)
        self.assertEqual(events[0].tracker_id, 11)

    def test_multiple_lines(self):
        self.engine.set_lines([self.line, CrossingLine(2, "Second", (10, -10), (10, 10))])
        events = self.feed(1, [(-5, 0), (5, 0), (6, 0), (15, 0), (16, 0)], start=0)
        self.assertEqual({e.line_id for e in events}, {1, 2})

    def test_person_and_car_simultaneously(self):
        tracks = {1: FakeTrack(1, 0, "person", .9, (-5, 0)), 2: FakeTrack(2, 2, "car", .9, (-4, 1))}
        self.engine.process_tracks(tracks, 0, 0, 30)
        tracks[1].pos, tracks[2].pos = (5, 0), (4, 1)
        self.engine.process_tracks(tracks, 1, 1, 30)
        tracks[1].pos, tracks[2].pos = (6, 0), (5, 1)
        events = self.engine.process_tracks(tracks, 2, 2, 30)
        self.assertEqual({e.class_name for e in events}, {"person", "car"})

    def test_temporary_class_change(self):
        self.feed(1, [(-5, 0), (-4, 0), (-3, 0)], class_id=2, name="car")
        events = self.feed(1, [(5, 0), (6, 0)], start=3, class_id=7, name="truck")
        self.assertEqual(events[0].class_name, "car")

    def test_robust_zone_a_line_zone_b(self):
        robust = CrossingLine(1, "Zoned", (0, -10), (0, 10), mode="robust",
                              zone_a=[(-10,-10),(-1,-10),(-1,10),(-10,10)],
                              zone_b=[(1,-10),(10,-10),(10,10),(1,10)])
        self.engine.set_lines([robust])
        self.assertEqual(len(self.feed(1, [(-5,0),(5,0),(6,0)])), 1)

    def test_diagonal_crossing(self):
        self.engine.set_lines([CrossingLine(1, "Diagonal", (-10,-10), (10,10))])
        self.assertEqual(len(self.feed(1, [(-5,5),(5,-5),(6,-6)])), 1)

    def test_segment_intersection_is_finite(self):
        self.assertIsNotNone(segment_intersection((100,100),(500,100),(300,0),(300,200)))
        self.assertIsNone(segment_intersection((100,300),(500,300),(300,0),(300,200)))

    def test_confirmed_event_updates_sqlite_and_intervals(self):
        event = self.feed(1, [(-5, 0), (5, 0), (6, 0)])[0]
        payload = event.to_dict()
        payload.update(track_id=event.tracker_id, class_label=event.class_name,
                       direction=event.direction, position_x=event.x, position_y=event.y)
        with TemporaryDirectory() as tmp:
            database = DatabaseManager(str(Path(tmp) / "v3.db"))
            session_id = database.create_session("test")
            database.add_crossing_event(session_id, payload)
            stored = database.get_session_events(session_id)
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0]['logical_track_id'], event.logical_track_id)
            self.assertEqual(stored[0]['event_id'], event.event_id)
        analytics = AnalyticsEngine()
        analytics.add_event(payload)
        self.assertEqual(analytics.get_volume_for_interval(1), 1)
        self.assertEqual(analytics.get_class_distribution()['car'], 1)


if __name__ == "__main__":
    unittest.main()
