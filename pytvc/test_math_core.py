import math
import unittest

import numpy as np

from pytvc.rigidBody import Quaternion, RigidBody, Vector3


RNG_SEED = 12345
RANDOM_CASES = 10_000


def assert_vector_close(
    testcase: unittest.TestCase,
    actual: Vector3,
    expected: Vector3,
    places: int = 9,
) -> None:
    testcase.assertAlmostEqual(actual.x, expected.x, places=places)
    testcase.assertAlmostEqual(actual.y, expected.y, places=places)
    testcase.assertAlmostEqual(actual.z, expected.z, places=places)


def assert_quat_close(
    testcase: unittest.TestCase,
    actual: Quaternion,
    expected: Quaternion,
    places: int = 9,
) -> None:
    testcase.assertAlmostEqual(actual.w, expected.w, places=places)
    testcase.assertAlmostEqual(actual.x, expected.x, places=places)
    testcase.assertAlmostEqual(actual.y, expected.y, places=places)
    testcase.assertAlmostEqual(actual.z, expected.z, places=places)


class TestVector3(unittest.TestCase):
    def test_construction_copy_and_mutation(self):
        v = Vector3(1.0, 2.0, 3.0)
        self.assertFalse(hasattr(v, "__dict__"))
        self.assertEqual((v.x, v.y, v.z), (1.0, 2.0, 3.0))

        c = v.copy()
        self.assertIsNot(c, v)
        self.assertEqual(c, v)

        self.assertIs(v.set(4.0, 5.0, 6.0), v)
        self.assertEqual(v, Vector3(4.0, 5.0, 6.0))
        self.assertIs(v.zero(), v)
        self.assertEqual(v, Vector3())
        self.assertIs(v.copy_from(c), v)
        self.assertEqual(v, c)

    def test_arithmetic_and_in_place_operations(self):
        a = Vector3(1.0, 2.0, 3.0)
        b = Vector3(4.0, -5.0, 6.0)

        self.assertEqual(a + b, Vector3(5.0, -3.0, 9.0))
        self.assertEqual(a - b, Vector3(-3.0, 7.0, -3.0))
        self.assertEqual(a * b, Vector3(4.0, -10.0, 18.0))
        self.assertEqual(a * 2.0, Vector3(2.0, 4.0, 6.0))
        self.assertEqual(3.0 * a, Vector3(3.0, 6.0, 9.0))
        self.assertEqual(b / Vector3(2.0, -5.0, 3.0), Vector3(2.0, 1.0, 2.0))
        self.assertEqual(b / 2.0, Vector3(2.0, -2.5, 3.0))

        c = a.copy()
        c += b
        self.assertEqual(c, Vector3(5.0, -3.0, 9.0))
        c -= b
        self.assertEqual(c, a)
        self.assertIs(c.add_scaled(b, 0.5), c)
        self.assertEqual(c, Vector3(3.0, -0.5, 6.0))
        self.assertIs(
            c.add_div_components(Vector3(8.0, 9.0, 10.0), Vector3(2.0, 3.0, 5.0)),
            c,
        )
        self.assertEqual(c, Vector3(7.0, 2.5, 8.0))

    def test_randomized_addition_multiplication_and_division(self):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(RANDOM_CASES):
            av = rng.uniform(-1_000.0, 1_000.0, size=3)
            bv = rng.uniform(-1_000.0, 1_000.0, size=3)
            bv = np.where(np.abs(bv) < 1e-9, 1.0, bv)
            scalar = float(rng.uniform(-100.0, 100.0))
            if abs(scalar) < 1e-9:
                scalar = 1.0

            a = Vector3(*av)
            b = Vector3(*bv)

            assert_vector_close(self, a + b, Vector3(*(av + bv)))
            assert_vector_close(self, a - b, Vector3(*(av - bv)))
            assert_vector_close(self, a * b, Vector3(*(av * bv)))
            assert_vector_close(self, a * scalar, Vector3(*(av * scalar)))
            assert_vector_close(self, scalar * a, Vector3(*(scalar * av)))
            assert_vector_close(self, a / b, Vector3(*(av / bv)))
            assert_vector_close(self, a / scalar, Vector3(*(av / scalar)))

    def test_iter_index_magnitude_norm_cross_dot_and_angle(self):
        v = Vector3(3.0, 4.0, 12.0)
        self.assertEqual(list(v), [3.0, 4.0, 12.0])
        self.assertEqual(v[0], 3.0)
        self.assertEqual(v[1], 4.0)
        self.assertEqual(v[2], 12.0)
        with self.assertRaises(IndexError):
            _ = v[3]

        self.assertEqual(v.mag2(), 169.0)
        self.assertEqual(abs(v), 13.0)
        self.assertEqual(v.len(), 13.0)
        assert_vector_close(self, v.norm(), Vector3(3.0 / 13.0, 4.0 / 13.0, 12.0 / 13.0))
        self.assertEqual(Vector3().norm(), Vector3())

        self.assertEqual(Vector3(1.0, 0.0, 0.0).dot(Vector3(0.0, 2.0, 0.0)), 0.0)
        self.assertEqual(
            Vector3(1.0, 0.0, 0.0).cross(Vector3(0.0, 1.0, 0.0)),
            Vector3(0.0, 0.0, 1.0),
        )
        self.assertAlmostEqual(
            Vector3(1.0, 0.0, 0.0).angleBetween(Vector3(0.0, 1.0, 0.0)),
            math.pi / 2.0,
        )
        self.assertEqual(Vector3().angleBetween(Vector3(1.0, 0.0, 0.0)), 0.0)

    def test_randomized_cross_dot_and_norm_against_numpy(self):
        rng = np.random.default_rng(RNG_SEED + 1)
        for _ in range(RANDOM_CASES):
            av = rng.uniform(-100.0, 100.0, size=3)
            bv = rng.uniform(-100.0, 100.0, size=3)
            a = Vector3(*av)
            b = Vector3(*bv)

            self.assertAlmostEqual(a.dot(b), float(np.dot(av, bv)))
            assert_vector_close(self, a.cross(b), Vector3(*np.cross(av, bv)))
            self.assertAlmostEqual(abs(a), float(np.linalg.norm(av)))
            if np.linalg.norm(av) > 0.0:
                assert_vector_close(self, a.norm(), Vector3(*(av / np.linalg.norm(av))))

    def test_repr_and_equality_type_check(self):
        v = Vector3(1.0, 2.0, 3.0)
        self.assertEqual(repr(v), "Vector3(1.0, 2.0, 3.0)")
        self.assertEqual(str(v), repr(v))
        self.assertNotEqual(v, object())
        self.assertFalse(v != Vector3(1.0, 2.0, 3.0))


class TestQuaternion(unittest.TestCase):
    def test_construction_copy_index_and_arithmetic(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        self.assertFalse(hasattr(q, "__dict__"))
        self.assertEqual(list(q), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(q[0], 1.0)
        self.assertEqual(q[3], 4.0)
        with self.assertRaises(IndexError):
            _ = q[4]

        self.assertEqual(q.copy(), q)
        self.assertIs(q.set(5.0, 6.0, 7.0, 8.0), q)
        self.assertEqual(q, Quaternion(5.0, 6.0, 7.0, 8.0))
        self.assertEqual(q + Quaternion(1.0, 1.0, 1.0, 1.0), Quaternion(6.0, 7.0, 8.0, 9.0))
        self.assertEqual(q - Quaternion(1.0, 2.0, 3.0, 4.0), Quaternion(4.0, 4.0, 4.0, 4.0))
        self.assertEqual(q * 2.0, Quaternion(10.0, 12.0, 14.0, 16.0))
        self.assertEqual(0.5 * q, Quaternion(2.5, 3.0, 3.5, 4.0))
        self.assertEqual(q / 2.0, Quaternion(2.5, 3.0, 3.5, 4.0))
        self.assertEqual(q / 0.0, Quaternion(0.0, 0.0, 0.0, 0.0))

    def test_hamilton_product_conjugate_norm_and_dot(self):
        a = Quaternion(1.0, 0.5, -0.25, 2.0)
        b = Quaternion(0.25, -1.0, 0.5, 0.75)
        self.assertEqual(a * b, Quaternion(-0.625, -2.0625, -1.9375, 1.25))
        self.assertEqual(a.conjugate(), Quaternion(1.0, -0.5, 0.25, -2.0))
        self.assertEqual(a.xyz, Vector3(0.5, -0.25, 2.0))
        self.assertAlmostEqual(a.dot(b), 1.125)
        self.assertAlmostEqual(abs(a), math.sqrt(a.mag2()))

        self.assertEqual(Quaternion(2.0, 0.0, 0.0, 0.0).norm(), Quaternion())
        q = Quaternion(0.0, 2.0, 0.0, 0.0)
        self.assertIs(q.normalize_ip(), q)
        self.assertEqual(q, Quaternion(0.0, 1.0, 0.0, 0.0))

    def test_rotation_helpers(self):
        q = Quaternion.fromEulerAngles(Vector3(0.0, math.pi / 2.0, 0.0))
        v = Vector3(1.0, 0.0, 0.0)
        assert_vector_close(self, q.rotate(v), Vector3(0.0, 0.0, -1.0))

        out = Vector3()
        self.assertIs(q.rotate_into(v, out), out)
        assert_vector_close(self, out, Vector3(0.0, 0.0, -1.0))

        self.assertIs(q.rotate_xyz_into(0.0, 0.0, 1.0, out), out)
        assert_vector_close(self, out, Vector3(1.0, 0.0, 0.0))

        unsafe = Quaternion(q.w * 10.0, q.x * 10.0, q.y * 10.0, q.z * 10.0)
        assert_vector_close(self, unsafe.rotateSafe(v), q.rotate(v))

    def test_randomized_rotations_preserve_vector_magnitude(self):
        rng = np.random.default_rng(RNG_SEED + 2)
        for _ in range(RANDOM_CASES):
            euler = Vector3(*rng.uniform(-math.pi, math.pi, size=3))
            raw = rng.uniform(-100.0, 100.0, size=3)
            v = Vector3(*raw)
            rotated = Quaternion.fromEulerAngles(euler).rotate(v)
            self.assertAlmostEqual(abs(rotated), float(np.linalg.norm(raw)), places=8)

    def test_axis_angle_euler_and_matrix_conversions(self):
        axis = Vector3(0.0, 0.0, 1.0)
        q = Quaternion.fromAxisAngle(axis, math.pi / 2.0)
        assert_vector_close(self, q.rotate(Vector3(1.0, 0.0, 0.0)), Vector3(0.0, 1.0, 0.0))

        recovered_axis, recovered_angle = q.toAxisAngle()
        assert_vector_close(self, recovered_axis, axis)
        self.assertAlmostEqual(recovered_angle, math.pi / 2.0)

        identity_axis, identity_angle = Quaternion().toAxisAngle()
        self.assertEqual(identity_axis, Vector3(1.0, 0.0, 0.0))
        self.assertEqual(identity_angle, 0.0)

        euler = Vector3(0.3, -0.4, 0.5)
        assert_vector_close(self, Quaternion.fromEulerAngles(euler).toEulerAngles(), euler)

        mat = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotated = Quaternion.fromRotationMatrix(mat).rotate(Vector3(1.0, 0.0, 0.0))
        assert_vector_close(self, rotated, Vector3(0.0, 1.0, 0.0))

        mat4 = np.eye(4)
        mat4[:3, :3] = mat
        rotated = Quaternion.fromRotationMatrix(mat4).rotate(Vector3(1.0, 0.0, 0.0))
        assert_vector_close(self, rotated, Vector3(0.0, 1.0, 0.0))

        with self.assertRaises(ValueError):
            Quaternion.fromRotationMatrix(np.eye(2))

    def test_repr_and_equality_type_check(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        self.assertEqual(repr(q), "Quaternion(1.0, 2.0, 3.0, 4.0)")
        self.assertEqual(str(q), repr(q))
        self.assertNotEqual(q, object())
        self.assertFalse(q != Quaternion(1.0, 2.0, 3.0, 4.0))


class TestRigidBody(unittest.TestCase):
    def make_body(self) -> RigidBody:
        return RigidBody(
            mass=2.0,
            inertia=Vector3(1.0, 2.0, 4.0),
            position=Vector3(0.0, 0.0, 0.0),
            velocity=Vector3(0.0, 0.0, 0.0),
            rotation=Quaternion(),
            rotVel=Vector3(0.0, 0.0, 0.0),
        )

    def test_initialization_normalizes_rotation_and_sets_inverse_mass(self):
        body = RigidBody(
            4.0,
            Vector3(1.0, 1.0, 1.0),
            Vector3(),
            Vector3(),
            Quaternion(2.0, 0.0, 0.0, 0.0),
            Vector3(),
        )
        self.assertEqual(body.mass, 4.0)
        self.assertEqual(body.inv_mass, 0.25)
        self.assertEqual(body.rotation, Quaternion())
        self.assertEqual(body.getAccel(), Vector3())

    def test_apply_force_integrates_linear_and_angular_state(self):
        body = self.make_body()
        body.applyForce(Vector3(4.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0))
        body.update(0.5)

        assert_vector_close(self, body.velocity, Vector3(1.0, 0.0, 0.0))
        assert_vector_close(self, body.position, Vector3(0.5, 0.0, 0.0))
        assert_vector_close(self, body.rotVel, Vector3(0.0, 0.0, -0.5))
        assert_vector_close(self, body.getAccel(), Vector3(2.0, 0.0, 0.0))
        self.assertEqual(body._accel, Vector3())
        self.assertEqual(body._torque, Vector3())

    def test_apply_local_force_rotates_force_and_torque(self):
        body = self.make_body()
        body.rotation = Quaternion.fromEulerAngles(Vector3(0.0, 0.0, math.pi / 2.0))
        body.applyLocalForce(Vector3(2.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0))
        body.update(1.0)

        assert_vector_close(self, body.velocity, Vector3(0.0, 1.0, 0.0))
        assert_vector_close(self, body.position, Vector3(0.0, 1.0, 0.0))
        assert_vector_close(self, body.rotVel, Vector3(0.0, 0.0, -0.5))

    def test_apply_torque_and_local_torque(self):
        body = self.make_body()
        body.applyTorque(Vector3(1.0, 2.0, 4.0))
        body.update(0.25)
        assert_vector_close(self, body.rotVel, Vector3(0.25, 0.25, 0.25))

        body = self.make_body()
        body.rotation = Quaternion.fromEulerAngles(Vector3(0.0, 0.0, math.pi / 2.0))
        body.applyLocalTorque(Vector3(2.0, 0.0, 0.0))
        body.update(1.0)
        assert_vector_close(self, body.rotVel, Vector3(0.0, 1.0, 0.0))

    def test_rotation_updates_from_existing_rotational_velocity(self):
        body = self.make_body()
        body.rotVel = Vector3(0.0, 0.0, math.pi)
        body.update(0.5)
        rotated = body.rotation.rotate(Vector3(1.0, 0.0, 0.0))
        assert_vector_close(self, rotated, Vector3(0.0, 1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
