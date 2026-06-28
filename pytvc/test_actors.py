import math
import unittest

from pytvc.actor import (
    Actor,
    Airbrake,
    AirbrakeCan,
    Fin,
    FinCan,
    MotorMount,
    RocketBody,
    RollDampener,
    SpinCan,
    TVCMount,
)
from pytvc.motor import Motor
from pytvc.rigidBody import Quaternion, RigidBody, Vector3
from pytvc.telemetry import Logger, SimClock


def assert_vector_close(
    testcase: unittest.TestCase,
    actual: Vector3,
    expected: Vector3,
    places: int = 8,
) -> None:
    testcase.assertAlmostEqual(actual.x, expected.x, places=places)
    testcase.assertAlmostEqual(actual.y, expected.y, places=places)
    testcase.assertAlmostEqual(actual.z, expected.z, places=places)


def make_body(
    *,
    position: Vector3 | None = None,
    velocity: Vector3 | None = None,
    rotation: Quaternion | None = None,
    rot_vel: Vector3 | None = None,
) -> RigidBody:
    return RigidBody(
        mass=1.0,
        inertia=Vector3(1.0, 1.0, 1.0),
        position=position or Vector3(10.0, 0.0, 0.0),
        velocity=velocity or Vector3(),
        rotation=rotation or Quaternion(),
        rotVel=rot_vel or Vector3(),
    )


def make_motor() -> Motor:
    motor = Motor.__new__(Motor)
    motor._points = [
        {"time": 0.0, "thrust": 0.0, "mass": 100.0},
        {"time": 1.0, "thrust": 10.0, "mass": 50.0},
        {"time": 2.0, "thrust": 0.0, "mass": 0.0},
    ]
    return motor


def lift_two(aoa: float, speed: float, state: RigidBody) -> float:
    return 2.0


def drag_half(aoa: float, speed: float, state: RigidBody) -> float:
    return 0.5


def drag_two(aoa: float, speed: float, state: RigidBody) -> float:
    return 2.0


def pressure_one_atm(state: RigidBody) -> float:
    return 1.225


def area_two(aoa: float, speed: float, state: RigidBody) -> float:
    return 2.0


class TestBaseActorAndRollDampener(unittest.TestCase):
    def test_base_actor_defaults_and_logging(self):
        actor = Actor()
        body = make_body()
        actor.update(body, 0.0)
        self.assertEqual(actor.getForce(), Vector3())
        self.assertEqual(actor.getTorque(), Vector3())
        self.assertEqual(actor.getMass(), 0.0)

        logger = Logger(SimClock())
        actor.logState(logger, prefix="base_")
        data = logger.toDict()
        for name in (
            "base_force_x",
            "base_force_y",
            "base_force_z",
            "base_torque_x",
            "base_torque_y",
            "base_torque_z",
            "base_mass",
        ):
            self.assertIn(name, data)

    def test_roll_dampener_validates_and_rotates_damping_torque(self):
        with self.assertRaises(TypeError):
            RollDampener((1.0, 2.0, 3.0))  # type: ignore[arg-type]

        body = make_body(rot_vel=Vector3(2.0, -3.0, 4.0))
        dampener = RollDampener(Vector3(-0.5, -1.0, -2.0))
        dampener.update(body, 0.0)
        assert_vector_close(self, dampener.getTorque(), Vector3(-1.0, 3.0, -8.0))

        logger = Logger(SimClock())
        dampener.logState(logger)
        self.assertIn("damping_torque_x", logger.toDict())


class TestMotorMounts(unittest.TestCase):
    def test_motor_mount_validation_ignition_force_mass_and_logging(self):
        motor = make_motor()
        mount = MotorMount(motor)
        with self.assertRaises(IndexError):
            mount.igniteMotor(1, 0.0)
        with self.assertRaises(ValueError):
            mount.igniteMotor(make_motor(), 0.0)

        mount.igniteMotor(0, 0.0)
        mount.update(make_body(), 0.5)
        assert_vector_close(self, mount.getForce(), Vector3(5.0, 0.0, 0.0))
        self.assertEqual(mount.getTorque(), Vector3())
        self.assertAlmostEqual(mount.getMass(), 0.075)

        logger = Logger(SimClock())
        mount.logState(logger)
        data = logger.toDict()
        self.assertEqual(data["thrust"]["values"], [5.0])
        self.assertEqual(data["motor_mass_g"]["values"], [75.0])
        self.assertEqual(data["motor0_ignited"]["values"], [True])

        with self.assertRaises(TypeError):
            MotorMount([motor, object()])  # type: ignore[list-item]

    def test_tvc_mount_callbacks_setpoint_force_angles_and_logging(self):
        calls: list[tuple[float, Vector3, float, float]] = []

        def servo(body: RigidBody, time: float, target: Vector3, thrust: float, mass: float) -> Vector3:
            calls.append((time, target, thrust, mass))
            return target * 0.5

        def linkage(body: RigidBody, time: float, servo_output: Vector3) -> Quaternion:
            return Quaternion.fromEulerAngles(servo_output)

        mount = TVCMount(make_motor(), servo, linkage)
        target = Vector3(0.0, math.pi / 2.0, 0.0)
        mount.setTargetAngles(target)
        mount.igniteMotor(0, 0.0)
        mount.update(make_body(), 1.0)

        self.assertEqual(mount.getSetpoint(), target)
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(calls[0][0], 1.0)
        self.assertEqual(calls[0][1], target)
        self.assertAlmostEqual(calls[0][2], 10.0)
        self.assertAlmostEqual(calls[0][3], 50.0)
        assert_vector_close(self, mount.getForce(), Vector3(10.0 / math.sqrt(2.0), 0.0, -10.0 / math.sqrt(2.0)))
        self.assertIsInstance(mount.getAngles(), Quaternion)

        logger = Logger(SimClock())
        mount.logState(logger)
        data = logger.toDict()
        self.assertIn("setpoint_y", data)
        self.assertIn("gimbal_pitch", data)

        with self.assertRaises(TypeError):
            TVCMount(make_motor(), object(), linkage)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            TVCMount(make_motor(), servo, object())  # type: ignore[arg-type]


class TestFins(unittest.TestCase):
    def test_aero_component_validation_exposed_by_fin(self):
        with self.assertRaises(ValueError):
            Fin((0.0, 0.0, 0.0), drag_half, lift_two, pressure_one_atm, 1.0, 0.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            Fin(Vector3(), object(), lift_two, pressure_one_atm, 1.0, 0.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            Fin(Vector3(), drag_half, object(), pressure_one_atm, 1.0, 0.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            Fin(Vector3(), drag_half, lift_two, object(), 1.0, 0.0)  # type: ignore[arg-type]

    def test_fin_angle_rotation_zero_flow_and_aero_forces(self):
        fin = Fin(Vector3(0.0, 1.0, 0.0), drag_half, lift_two, pressure_one_atm, 1.0, 0.0)
        fin.setAngle(0.25)
        self.assertEqual(fin.getAngle(), 0.25)
        self.assertEqual(fin.getBodyAngle(), 0.0)
        self.assertIsInstance(fin.getRotationBody(), Quaternion)

        fin.update(make_body(velocity=Vector3()), 0.0)
        self.assertEqual(fin.getForce(), Vector3())
        self.assertEqual(fin.getTorque(), Vector3())

        fin.setAngle(0.0)
        fin.update(make_body(velocity=Vector3(10.0, 0.0, 0.0)), 0.0)
        assert_vector_close(self, fin.getForce(), Vector3(-30.625, 0.0, 122.5))
        assert_vector_close(self, fin.liftForces[-1], Vector3(0.0, 0.0, 122.5))
        assert_vector_close(self, fin.dragForces[-1], Vector3(-30.625, 0.0, 0.0))

        logger = Logger(SimClock())
        fin.logState(logger)
        data = logger.toDict()
        self.assertEqual(data["angle"]["values"], [0.0])
        self.assertIn("lift_z", data)
        self.assertIn("drag_x", data)

    def test_fin_can_constructs_updates_and_sums_child_forces(self):
        can = FinCan(
            Vector3(1.0, 0.0, 0.0),
            drag_half,
            lift_two,
            pressure_one_atm,
            area=1.0,
            radialDistance=0.5,
            finCount=4,
            offsetInitializer=lambda: 0.1,
        )
        self.assertEqual(len(can.getFins()), 4)
        self.assertTrue(all(abs(fin.getAngle() - 0.1) < 1e-12 for fin in can.getFins()))
        self.assertEqual(can.position, Vector3(1.0, 0.0, 0.0))

        can.update(make_body(velocity=Vector3(10.0, 0.0, 0.0)), 0.0)
        expected_force = Vector3()
        for fin in can.getFins():
            expected_force += fin.getForce()
        assert_vector_close(self, can.getForce(), expected_force)

        expected_torque = Vector3()
        for fin in can.getFins():
            expected_torque += fin.getTorque()
            expected_torque += fin.position.cross(fin.getForce()) * Vector3(1.0, 0.0, 0.0)
        assert_vector_close(self, can.getTorque(), expected_torque)

        logger = Logger(SimClock())
        can.logState(logger)
        self.assertIn("fin0_lift_x", logger.toDict())

    def test_spin_can_applies_roll_coefficient_to_fin_force_torque(self):
        can = SpinCan(
            Vector3(),
            drag_half,
            lift_two,
            pressure_one_atm,
            area=1.0,
            radialDistance=1.0,
            finCount=2,
            rollCoefficient=0.25,
        )
        for fin in can.getFins():
            fin._Fin__force = Vector3(0.0, 0.0, 2.0)

        expected = Vector3()
        for fin in can.getFins():
            expected += fin.position.cross(fin.getForce()) * Vector3(0.25, 0.0, 0.0)
        assert_vector_close(self, can.getTorque(), expected)


class TestAirbrakesAndRocketBody(unittest.TestCase):
    def test_airbrake_angle_clip_zero_flow_drag_and_logging(self):
        brake = Airbrake(Vector3(0.0, 1.0, 0.0), 0.0, 1.0, drag_two, maxAngle=math.pi / 2.0)
        with self.assertRaises(TypeError):
            Airbrake((0.0, 0.0, 0.0), 0.0, 1.0, drag_two)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            Airbrake(Vector3(), 0.0, 1.0, object())  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            Airbrake(Vector3(), 0.0, 0.0, drag_two)
        with self.assertRaises(ValueError):
            Airbrake(Vector3(), 0.0, 1.0, drag_two, maxAngle=0.0)

        brake.setAngle(math.pi)
        self.assertEqual(brake.getAngle(), math.pi / 2.0)
        self.assertEqual(brake.getBodyAngle(), 0.0)

        brake.update(make_body(velocity=Vector3()), 0.0)
        self.assertEqual(brake.getForce(), Vector3())
        self.assertEqual(brake.getTorque(), Vector3())
        self.assertEqual(brake.getLastDrag(), Vector3())

        brake.update(make_body(velocity=Vector3(10.0, 0.0, 0.0)), 0.0)
        assert_vector_close(self, brake.getForce(), Vector3(-122.5, 0.0, 0.0))
        assert_vector_close(self, brake.getLastDrag(), Vector3(-122.5, 0.0, 0.0))

        logger = Logger(SimClock())
        brake.logState(logger)
        self.assertIn("last_drag_x", logger.toDict())

    def test_airbrake_can_constructs_clips_updates_and_sums(self):
        with self.assertRaises(ValueError):
            AirbrakeCan(Vector3(), drag_two, 1.0, 1.0, 0)

        can = AirbrakeCan(Vector3(), drag_two, area=1.0, radialDistance=1.0, brakeCount=3)
        self.assertEqual(len(can.getAirbrakes()), 3)
        can.setAngle(math.pi / 4.0)
        self.assertTrue(all(abs(brake.getAngle() - math.pi / 4.0) < 1e-12 for brake in can.getAirbrakes()))

        can.update(make_body(velocity=Vector3(10.0, 0.0, 0.0)), 0.0)
        expected_force = Vector3()
        expected_torque = Vector3()
        for brake in can.getAirbrakes():
            expected_force += brake.getForce()
            expected_torque += brake.position.cross(brake.getForce())
        assert_vector_close(self, can.getForce(), expected_force)
        assert_vector_close(self, can.getTorque(), expected_torque)
        self.assertEqual(len(can.getDragVectors()), 3)

        logger = Logger(SimClock())
        can.logState(logger)
        self.assertIn("brake0_last_drag_x", logger.toDict())

    def test_rocket_body_aero_force_and_validation(self):
        with self.assertRaises(TypeError):
            RocketBody(Vector3(), drag_half, lift_two, pressure_one_atm, object())  # type: ignore[arg-type]

        body_actor = RocketBody(Vector3(1.0, 0.0, 0.0), drag_half, lift_two, pressure_one_atm, area_two)
        body = make_body(velocity=Vector3(0.0, 10.0, 0.0))
        body_actor.update(body, 0.0)

        assert_vector_close(self, body_actor.getLift(), Vector3(0.0, -245.0, 0.0))
        assert_vector_close(self, body_actor.getDrag(), Vector3(0.0, -61.25, 0.0))
        assert_vector_close(self, body_actor.getForce(), Vector3(0.0, -306.25, 0.0))
        self.assertEqual(body_actor.getTorque(), Vector3())

        body_actor.update(make_body(velocity=Vector3()), 0.0)
        self.assertEqual(body_actor.getForce(), Vector3())

        logger = Logger(SimClock())
        body_actor.logState(logger)
        self.assertIn("lift_x", logger.toDict())


if __name__ == "__main__":
    unittest.main()
