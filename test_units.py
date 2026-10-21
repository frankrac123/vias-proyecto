"""
Unit tests for traffic detection system
Tests critical functions: line intersection, direction detection, counting, PHF calculations
"""
import unittest
from datetime import datetime, timedelta
from app.lines_manager import CountingEvent, CountingLine, LinesManager, TrafficFlowStats


class TestCountingLine(unittest.TestCase):
    """Test CountingLine geometry"""
    
    def test_pixel_coordinates_conversion(self):
        """Test normalized to pixel coordinate conversion"""
        line = CountingLine(
            id=1,
            name="Test",
            norm_x1=0.0,
            norm_y1=0.5,
            norm_x2=1.0,
            norm_y2=0.5,
            direction_a="A",
            direction_b="B",
            frame_height=480,
            frame_width=640
        )
        
        x1, y1, x2, y2 = line.get_pixel_coords()
        
        self.assertEqual(x1, 0)
        self.assertEqual(y1, 240)
        self.assertEqual(x2, 640)
        self.assertEqual(y2, 240)
    
    def test_line_serialization(self):
        """Test line to_dict conversion"""
        line = CountingLine(
            id=1,
            name="Test Line",
            norm_x1=0.1,
            norm_y1=0.2,
            norm_x2=0.9,
            norm_y2=0.8,
            direction_a="Norte",
            direction_b="Sur",
        )
        
        line_dict = line.to_dict()
        
        self.assertEqual(line_dict['name'], "Test Line")
        self.assertEqual(line_dict['norm_x1'], 0.1)
        self.assertEqual(line_dict['direction_a'], "Norte")


class TestTrafficFlowStats(unittest.TestCase):
    """Test traffic statistics calculations"""
    
    def test_flow_rate_calculation_veh_per_hour(self):
        """Test conversion: veh/min -> veh/h"""
        stats = TrafficFlowStats(
            interval_minutes=15,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=15),
            total_count=25
        )
        
        stats.calculate_flow_rates()
        
        # 25 vehicles in 15 minutes = 100 veh/h
        self.assertAlmostEqual(stats.flow_rate_per_hour, 100.0, places=1)
        self.assertAlmostEqual(stats.flow_rate_per_min, 25/15, places=2)
    
    def test_flow_rate_calculation_1_minute(self):
        """Test 1-minute interval calculation"""
        stats = TrafficFlowStats(
            interval_minutes=1,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=1),
            total_count=50
        )
        
        stats.calculate_flow_rates()
        
        # 50 vehicles in 1 minute = 3000 veh/h
        self.assertAlmostEqual(stats.flow_rate_per_hour, 3000.0, places=1)
        self.assertEqual(stats.flow_rate_per_min, 50.0)
    
    def test_class_distribution(self):
        """Test vehicle class distribution percentages"""
        stats = TrafficFlowStats(
            interval_minutes=1,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=1),
            total_count=100
        )
        
        stats.cars = 50
        stats.motorcycles = 30
        stats.buses = 20
        
        distribution = stats.get_class_distribution()
        
        self.assertEqual(distribution['Carros'], 50.0)
        self.assertEqual(distribution['Motos'], 30.0)
        self.assertEqual(distribution['Buses'], 20.0)
    
    def test_zero_division_flow_rate(self):
        """Test flow rate with zero elapsed time"""
        stats = TrafficFlowStats(
            interval_minutes=0,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_count=0
        )
        
        # Should not raise exception
        stats.calculate_flow_rates()
        
        self.assertEqual(stats.flow_rate_per_hour, 0.0)
        self.assertEqual(stats.flow_rate_per_min, 0.0)


class TestLinesManager(unittest.TestCase):
    """Test LinesManager functionality"""
    
    def setUp(self):
        """Create fresh manager for each test"""
        self.manager = LinesManager()
        # Don't load from config during tests
        self.manager.lines.clear()
        self.manager.stats.clear()
        self.manager.events.clear()
        self.manager.next_line_id = 1
    
    def test_add_line(self):
        """Test adding a counting line"""
        line = self.manager.add_line(
            name="Test Line",
            norm_x1=0.0,
            norm_y1=0.5,
            norm_x2=1.0,
            norm_y2=0.5,
            direction_a="A",
            direction_b="B"
        )
        
        self.assertEqual(line.id, 1)
        self.assertEqual(line.name, "Test Line")
        self.assertIn(1, self.manager.lines)
    
    def test_remove_line(self):
        """Test removing a counting line"""
        self.manager.add_line(
            name="Test",
            norm_x1=0.0,
            norm_y1=0.5,
            norm_x2=1.0,
            norm_y2=0.5
        )
        
        self.assertTrue(self.manager.remove_line(1))
        self.assertNotIn(1, self.manager.lines)
    
    def test_record_crossing(self):
        """Test recording a crossing event"""
        self.manager.add_line(
            name="Test",
            norm_x1=0.0,
            norm_y1=0.5,
            norm_x2=1.0,
            norm_y2=0.5
        )
        
        event = CountingEvent(
            timestamp=datetime.now(),
            track_id=1,
            class_id=2,
            class_label="Carro",
            confidence=0.95,
            direction="A->B",
            line_id=1,
            position_x=0.5,
            position_y=0.5
        )
        
        self.assertTrue(self.manager.record_crossing(event))
        self.assertEqual(len(self.manager.events), 1)
    
    def test_duplicate_crossing_prevention(self):
        """Test that duplicate crossings within 0.5s are rejected"""
        self.manager.add_line(
            name="Test",
            norm_x1=0.0,
            norm_y1=0.5,
            norm_x2=1.0,
            norm_y2=0.5
        )
        
        now = datetime.now()
        
        event1 = CountingEvent(
            timestamp=now,
            track_id=1,
            class_id=2,
            class_label="Carro",
            confidence=0.95,
            direction="A->B",
            line_id=1,
            position_x=0.5,
            position_y=0.5
        )
        
        event2 = CountingEvent(
            timestamp=now + timedelta(milliseconds=100),
            track_id=1,
            class_id=2,
            class_label="Carro",
            confidence=0.95,
            direction="A->B",
            line_id=1,
            position_x=0.5,
            position_y=0.5
        )
        
        # First crossing should be accepted
        self.assertTrue(self.manager.record_crossing(event1))
        
        # Duplicate within 0.5s should be rejected
        self.assertFalse(self.manager.record_crossing(event2))
        
        # Only one event recorded
        self.assertEqual(len(self.manager.events), 1)
    
    def test_get_volume(self):
        """Test volume calculation"""
        self.manager.add_line(
            name="Test",
            norm_x1=0.0,
            norm_y1=0.5,
            norm_x2=1.0,
            norm_y2=0.5
        )
        
        now = datetime.now()
        
        # Record 5 crossings
        for i in range(5):
            event = CountingEvent(
                timestamp=now + timedelta(seconds=i*5),
                track_id=i,
                class_id=2,
                class_label="Carro",
                confidence=0.95,
                direction="A->B" if i % 2 == 0 else "B->A",
                line_id=1,
                position_x=0.5,
                position_y=0.5
            )
            self.manager.record_crossing(event)
        
        volume = self.manager.get_volume(1, interval_minutes=1)
        self.assertEqual(volume, 5)
    
    def test_flow_rate_per_hour(self):
        """Test flow rate calculation in veh/h"""
        self.manager.add_line(
            name="Test",
            norm_x1=0.0,
            norm_y1=0.5,
            norm_x2=1.0,
            norm_y2=0.5
        )
        
        now = datetime.now()
        
        # Record 25 crossings in 15 minutes
        for i in range(25):
            event = CountingEvent(
                timestamp=now + timedelta(seconds=i*36),  # Spread over ~15 minutes
                track_id=i,
                class_id=2,
                class_label="Carro",
                confidence=0.95,
                direction="A->B",
                line_id=1,
                position_x=0.5,
                position_y=0.5
            )
            self.manager.record_crossing(event)
        
        # In 15 minute interval: 25 vehicles = 100 veh/h
        flow_rate = self.manager.get_flow_rate(1, interval_minutes=15, unit="veh/h")
        self.assertGreater(flow_rate, 0)  # Should have some flow rate
    
    def test_direction_split(self):
        """Test directional counting"""
        self.manager.add_line(
            name="Test",
            norm_x1=0.0,
            norm_y1=0.5,
            norm_x2=1.0,
            norm_y2=0.5
        )
        
        now = datetime.now()
        
        # Record 3 A->B and 2 B->A
        for i in range(3):
            event = CountingEvent(
                timestamp=now + timedelta(seconds=i),
                track_id=i,
                class_id=2,
                class_label="Carro",
                confidence=0.95,
                direction="A->B",
                line_id=1,
                position_x=0.5,
                position_y=0.5
            )
            self.manager.record_crossing(event)
        
        for i in range(3, 5):
            event = CountingEvent(
                timestamp=now + timedelta(seconds=i),
                track_id=i,
                class_id=2,
                class_label="Carro",
                confidence=0.95,
                direction="B->A",
                line_id=1,
                position_x=0.5,
                position_y=0.5
            )
            self.manager.record_crossing(event)
        
        stats = self.manager.get_stats(1, interval_minutes=1)
        
        self.assertEqual(stats.direction_ab, 3)
        self.assertEqual(stats.direction_ba, 2)
        self.assertEqual(stats.total_count, 5)
    
    def test_class_counting(self):
        """Test counting by vehicle class"""
        self.manager.add_line(
            name="Test",
            norm_x1=0.0,
            norm_y1=0.5,
            norm_x2=1.0,
            norm_y2=0.5
        )
        
        now = datetime.now()
        
        # Record mixed classes
        classes = [
            (2, "Carro", 3),      # 3 cars
            (3, "Moto", 2),       # 2 motorcycles
            (0, "Persona", 1),    # 1 person
        ]
        
        event_id = 0
        for class_id, label, count in classes:
            for i in range(count):
                event = CountingEvent(
                    timestamp=now + timedelta(seconds=event_id),
                    track_id=event_id,
                    class_id=class_id,
                    class_label=label,
                    confidence=0.95,
                    direction="A->B",
                    line_id=1,
                    position_x=0.5,
                    position_y=0.5
                )
                self.manager.record_crossing(event)
                event_id += 1
        
        stats = self.manager.get_stats(1, interval_minutes=1)
        
        self.assertEqual(stats.cars, 3)
        self.assertEqual(stats.motorcycles, 2)
        self.assertEqual(stats.persons, 1)
        self.assertEqual(stats.vehicles_count, 5)
        self.assertEqual(stats.pedestrians_count, 1)


class TestGeometricIntersection(unittest.TestCase):
    """Test geometric line intersection for fast vehicles"""
    
    @staticmethod
    def ccw(A, B, C):
        """Check if three points are in counter-clockwise order"""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    @staticmethod
    def segments_intersect(A, B, C, D):
        """Check if line segment AB intersects with line segment CD"""
        return TestGeometricIntersection.ccw(A, C, D) != TestGeometricIntersection.ccw(B, C, D) and \
               TestGeometricIntersection.ccw(A, B, C) != TestGeometricIntersection.ccw(A, B, D)
    
    def test_horizontal_line_crossing(self):
        """Test detection of crossing horizontal line"""
        # Line from (0, 100) to (100, 100)
        line_start = (0, 100)
        line_end = (100, 100)
        
        # Object crosses from top to bottom: (50, 50) -> (50, 150)
        obj_prev = (50, 50)
        obj_curr = (50, 150)
        
        self.assertTrue(self.segments_intersect(line_start, line_end, obj_prev, obj_curr))
    
    def test_vertical_line_crossing(self):
        """Test detection of crossing vertical line"""
        # Line from (50, 0) to (50, 100)
        line_start = (50, 0)
        line_end = (50, 100)
        
        # Object crosses from left to right: (0, 50) -> (100, 50)
        obj_prev = (0, 50)
        obj_curr = (100, 50)
        
        self.assertTrue(self.segments_intersect(line_start, line_end, obj_prev, obj_curr))
    
    def test_no_crossing(self):
        """Test when object doesn't cross line"""
        # Line from (0, 100) to (100, 100)
        line_start = (0, 100)
        line_end = (100, 100)
        
        # Object stays above line: (50, 0) -> (60, 10)
        obj_prev = (50, 0)
        obj_curr = (60, 10)
        
        self.assertFalse(self.segments_intersect(line_start, line_end, obj_prev, obj_curr))
    
    def test_fast_vehicle_jump(self):
        """Test fast vehicle jumping over line between frames"""
        # Line from (0, 100) to (100, 100)
        line_start = (0, 100)
        line_end = (100, 100)
        
        # Fast object jumps over line: (50, 50) -> (50, 150)
        obj_prev = (50, 50)
        obj_curr = (50, 150)
        
        # Should detect crossing even though vehicle jumped
        self.assertTrue(self.segments_intersect(line_start, line_end, obj_prev, obj_curr))


def run_tests():
    """Run all unit tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestCountingLine))
    suite.addTests(loader.loadTestsFromTestCase(TestTrafficFlowStats))
    suite.addTests(loader.loadTestsFromTestCase(TestLinesManager))
    suite.addTests(loader.loadTestsFromTestCase(TestGeometricIntersection))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
