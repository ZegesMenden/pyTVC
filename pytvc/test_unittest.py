from pytvc.physics import *
import numpy as np
import unittest

DEG_TO_RAD = np.pi / 180

class test_vector_math(unittest.TestCase):

    def test_eq(self):
        self.assertEqual(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0))

    def test_add(self):
        self.assertEqual(vec3(0.0, 0.0, 0.0) +
                         vec3(1.0, 1.0, 1.0), vec3(1.0, 1.0, 1.0))

    def test_sub(self):
        self.assertEqual(vec3(0.0, 0.0, 0.0) - vec3(1.0,
                         1.0, 1.0), vec3(-1.0, -1.0, -1.0))

    def test_mul(self):
        self.assertEqual(vec3(1.0, 1.0, 1.0) *
                         vec3(2.0, 2.0, 2.0), vec3(2.0, 2.0, 2.0))

    def test_mul_scalar(self):
        self.assertEqual(vec3(1.0, 1.0, 1.0) * 2.0, vec3(2.0, 2.0, 2.0))

    def test_div(self):
        self.assertEqual(vec3(1.0, 1.0, 1.0) /
                         vec3(2.0, 2.0, 2.0), vec3(0.5, 0.5, 0.5))

    def test_div_scalar(self):
        self.assertEqual(vec3(1.0, 1.0, 1.0) / 2.0, vec3(0.5, 0.5, 0.5))

    def test_norm(self):
        self.assertEqual(vec3(5.0, 5.0, 5.0).normalize(), vec3(
            0.5773502691896257, 0.5773502691896257, 0.5773502691896257))

    def test_len(self):
        self.assertEqual(vec3(5.0, 5.0, 5.0).length(), 8.660254037844387)

    def test_abs(self):
        self.assertEqual(vec3(1.0, 1.0, 1.0),
                         abs(vec3(-1.0, -1.0, -1.0)))

    def test_round(self):
        self.assertEqual(vec3(1.1, 1.1, 1.1),
                         round(vec3(1.11, 1.11, 1.11), 1))

    def test_str(self):
        self.assertEqual("1.0, 1.0, 1.0", str(vec3(1.0, 1.0, 1.0)))


class test_quaternion_math(unittest.TestCase):

    def test_eq(self):
        self.assertEqual(quaternion(1.0, 0.0, 0.0, 0.0),
                         quaternion(1.0, 0.0, 0.0, 0.0))

    def test_mul(self):
        self.assertEqual(quaternion(1.0, 0.5, 0.5, 0.5) * quaternion(1.0,
                         0.5, 0.5, 0.5), quaternion(0.25, 1.0, 1.0, 1.0))

    def test_conj(self):
        self.assertEqual(quaternion(1.0, 0.5, 0.5, 0.5).conjugate(),
                         quaternion(1.0, -0.5, -0.5, -0.5))

    def test_norm(self):
        q = quaternion(2.0, 2.0, 2.0, 2.0)
        self.assertAlmostEqual(q.length(), 4.0, 4)

    def test_rotate(self):

        q: quaternion = quaternion().from_euler(vec3(0, 90 * DEG_TO_RAD, 0))
        print(q)
        v: vec3 = vec3(1, 0, 0)

        vt = q.rotate(v)
        print(vt)

        self.assertAlmostEqual(vt.x, 0.0, 4)
        self.assertAlmostEqual(vt.y, 0.0, 4)
        self.assertAlmostEqual(vt.z, -1.0, 4)

    def test_euler_to_quaternion(self):

        e = vec3(45 * DEG_TO_RAD, 45 * DEG_TO_RAD, 45 * DEG_TO_RAD)
        q = quaternion().from_euler(e)

        qt = quaternion(0.8446231020115715, 0.19134170284356303,
                        0.4619399539487806, 0.19134170284356303)

        self.assertAlmostEqual(q.w, qt.w, 4)
        self.assertAlmostEqual(q.x, qt.x, 4)
        self.assertAlmostEqual(q.y, qt.y, 4)
        self.assertAlmostEqual(q.z, qt.z, 4)

    def test_quaternion_to_euler(self):

        q = quaternion(0.8446231020115715, 0.19134170284356303,
                       0.4619399539487806, 0.19134170284356303)

        e = q.to_euler()
        et = vec3(45 * DEG_TO_RAD, 45 * DEG_TO_RAD, 45 * DEG_TO_RAD)
        # print(q)
        # print(e * RAD_TO_DEG)

        self.assertAlmostEqual(e.x, et.x, 4)
        self.assertAlmostEqual(e.y, et.y, 4)
        self.assertAlmostEqual(e.z, et.z, 4)


if __name__ == '__main__':
    unittest.main()
