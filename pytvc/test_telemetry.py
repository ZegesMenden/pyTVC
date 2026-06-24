import os
import csv
import tempfile
import unittest

from pytvc.telemetry import SimClock, Logger, LogFormat
from pytvc.rigidBody import Vector3, Quaternion
from pytvc.rocket import Rocket
from pytvc.actor import RollDampener


class test_sim_clock(unittest.TestCase):

    def test_advance(self):
        c = SimClock()
        self.assertEqual(c.simTime, 0.0)
        c.advance(0.1)
        c.advance(0.2)
        self.assertAlmostEqual(c.simTime, 0.3)

    def test_advance_rejects_nonpositive(self):
        c = SimClock()
        with self.assertRaises(ValueError):
            c.advance(0.0)
        with self.assertRaises(ValueError):
            c.advance(-1.0)

    def test_reset(self):
        c = SimClock()
        c.advance(1.0)
        c.reset()
        self.assertEqual(c.simTime, 0.0)


class test_logger(unittest.TestCase):

    def test_log_auto_infers_format(self):
        lg = Logger(SimClock(), "a")
        lg.logAuto("n", 1.5)
        lg.logAuto("b", True)
        lg.logAuto("s", "hi")
        d = lg.toDict()
        self.assertEqual(d["n"]["format"], "NUMBER")
        self.assertEqual(d["b"]["format"], "BOOL")
        self.assertEqual(d["s"]["format"], "STRING")
        # bool must not be logged as a number
        self.assertEqual(d["b"]["values"], [True])

    def test_log_vector_and_quaternion(self):
        lg = Logger(SimClock(), "a")
        lg.logVector("v", Vector3(1, 2, 3))
        lg.logQuaternion("q", Quaternion(1, 0, 0, 0))
        names = lg.traceNames()
        for suffix in ("v_x", "v_y", "v_z", "q_w", "q_x", "q_y", "q_z"):
            self.assertIn(suffix, names)

    def test_merge_namespacing(self):
        c = SimClock()
        a = Logger(c, "a")
        b = Logger(c, "b")
        a.logScalar("foo", 1.0)
        b.logScalar("foo", 2.0)
        merged = Logger(c, "m")
        merged.merge(a)                 # top level
        merged.merge(b, prefix="bee")   # namespaced
        names = merged.traceNames()
        self.assertIn("foo", names)
        self.assertIn("bee/foo", names)

    def test_merge_collision_raises(self):
        c = SimClock()
        a = Logger(c, "a")
        b = Logger(c, "b")
        a.logScalar("foo", 1.0)
        b.logScalar("foo", 2.0)
        merged = Logger(c, "m")
        merged.merge(a)
        with self.assertRaises(ValueError):
            merged.merge(b)  # both expose top-level "foo"


class test_rocket_logging(unittest.TestCase):

    def _make_rocket(self) -> Rocket:
        r = Rocket(
            1.0,
            Vector3(1, 1, 1),
            Vector3(10, 0, 0),
            Vector3(0, 0, 0),
            Quaternion(),
            Vector3(0.1, 0, 0),
        )
        r.addActor(RollDampener(Vector3(0.5, 0.5, 0.5)), name="damp")
        return r

    def test_time_delegates_to_clock(self):
        r = self._make_rocket()
        r.update(0.01)
        r.update(0.01)
        self.assertAlmostEqual(r.time(), 0.02)
        self.assertIs(r.getClock(), r.getClock())

    def test_no_logging_when_disabled(self):
        r = self._make_rocket()
        for _ in range(3):
            r.update(0.01)
        self.assertEqual(r.getLog().traceNames(), [])

    def test_auto_record_and_merge(self):
        r = self._make_rocket()
        r.enableLogging()
        n = 5
        for _ in range(n):
            r.update(0.01)
        log = r.getLog()
        names = log.traceNames()
        self.assertIn("position_x", names)            # rocket body, top level
        self.assertIn("damp/damping_torque_x", names)  # actor, namespaced
        d = log.toDict()
        self.assertEqual(len(d["position_x"]["values"]), n)
        self.assertEqual(len(d["damp/damping_torque_x"]["values"]), n)

    def test_write_csv(self):
        r = self._make_rocket()
        r.enableLogging()
        n = 4
        for _ in range(n):
            r.update(0.01)
        path = os.path.join(tempfile.gettempdir(), "pytvc_tele_test.csv")
        r.getLog().writeCSV(path)
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows), n + 1)  # header + one row per step
        self.assertEqual(rows[0][0], "sim_time")
        self.assertEqual(rows[0][1], "real_time")


if __name__ == "__main__":
    unittest.main()
