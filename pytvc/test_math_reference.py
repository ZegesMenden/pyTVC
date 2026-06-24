"""Cross-check pyTVC's Vector3 / Quaternion math against trusted references.

Vector operations are compared against NumPy (``np.cross``, ``np.dot``,
``np.linalg.norm`` ...) and quaternion / rotation operations against SciPy's
``scipy.spatial.transform.Rotation``.

Conventions (verified empirically, see asserts below):
- pyTVC stores quaternions scalar-first ``(w, x, y, z)``; SciPy uses scalar-last
  ``(x, y, z, w)``.
- ``Quaternion.fromEulerAngles`` / ``toEulerAngles`` match SciPy's extrinsic
  ``"xyz"`` sequence (roll=x, pitch=y, yaw=z).
- ``q1 * q2`` composes like ``R1 * R2`` (the right-hand rotation is applied
  first).
"""

import unittest

import numpy as np

try:
    from scipy.spatial.transform import Rotation
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - scipy is a test-only dependency
    _HAVE_SCIPY = False

from pytvc.rigidBody import Vector3, Quaternion

ATOL = 1e-9


def _v(vec: Vector3) -> np.ndarray:
    return np.array([vec.x, vec.y, vec.z], dtype=float)


def _q_to_scipy(q: Quaternion) -> "Rotation":
    """pyTVC (w, x, y, z) -> SciPy Rotation (scalar-last quat)."""
    return Rotation.from_quat([q.x, q.y, q.z, q.w])


# Deterministic sample data reused across tests.
_VEC_PAIRS = [
    (Vector3(1.0, 2.0, 3.0), Vector3(-4.0, 5.0, -6.0)),
    (Vector3(0.0, 0.0, 1.0), Vector3(0.0, 1.0, 0.0)),
    (Vector3(-1.5, 0.25, 7.0), Vector3(3.0, -2.0, 0.5)),
    (Vector3(2.0, 2.0, 2.0), Vector3(2.0, 2.0, 2.0)),
]

_EULERS = [
    (0.3, -0.5, 0.9),
    (-0.1, 0.4, 0.2),
    (1.2, -1.0, 0.7),
    (0.0, 0.0, 0.0),
    (np.pi / 4, np.pi / 6, -np.pi / 3),
]


class test_vector_vs_numpy(unittest.TestCase):

    def test_add_sub_mul_div(self):
        for a, b in _VEC_PAIRS:
            na, nb = _v(a), _v(b)
            np.testing.assert_allclose(_v(a + b), na + nb, atol=ATOL)
            np.testing.assert_allclose(_v(a - b), na - nb, atol=ATOL)
            np.testing.assert_allclose(_v(a * b), na * nb, atol=ATOL)  # elementwise
            np.testing.assert_allclose(_v(a * 2.5), na * 2.5, atol=ATOL)

    def test_dot(self):
        for a, b in _VEC_PAIRS:
            self.assertAlmostEqual(a.dot(b), float(np.dot(_v(a), _v(b))), places=9)

    def test_cross(self):
        for a, b in _VEC_PAIRS:
            np.testing.assert_allclose(_v(a.cross(b)), np.cross(_v(a), _v(b)), atol=ATOL)

    def test_magnitude(self):
        for a, _ in _VEC_PAIRS:
            ref = float(np.linalg.norm(_v(a)))
            self.assertAlmostEqual(abs(a), ref, places=9)
            self.assertAlmostEqual(a.len(), ref, places=9)

    def test_normalize(self):
        for a, _ in _VEC_PAIRS:
            n = _v(a)
            mag = np.linalg.norm(n)
            if mag == 0.0:
                continue
            np.testing.assert_allclose(_v(a.norm()), n / mag, atol=ATOL)

    def test_angle_between(self):
        for a, b in _VEC_PAIRS:
            na, nb = _v(a), _v(b)
            denom = np.linalg.norm(na) * np.linalg.norm(nb)
            if denom == 0.0:
                continue
            cos = np.clip(np.dot(na, nb) / denom, -1.0, 1.0)
            self.assertAlmostEqual(a.angleBetween(b), float(np.arccos(cos)), places=9)


@unittest.skipUnless(_HAVE_SCIPY, "scipy is required for rotation reference tests")
class test_quaternion_vs_scipy(unittest.TestCase):

    _SAMPLE_VECS = [_v(a) for a, _ in _VEC_PAIRS]

    def test_from_euler_matches_xyz(self):
        for roll, pitch, yaw in _EULERS:
            q = Quaternion.fromEulerAngles(Vector3(roll, pitch, yaw))
            ref = Rotation.from_euler("xyz", [roll, pitch, yaw])
            for v in self._SAMPLE_VECS:
                rotated = _v(q.rotate(Vector3(*v)))
                np.testing.assert_allclose(rotated, ref.apply(v), atol=ATOL)

    def test_to_euler_matches_xyz(self):
        for rotvec in ([0.4, 0.2, -0.6], [1.0, -0.3, 0.8], [0.0, 0.0, 0.0]):
            ref = Rotation.from_rotvec(rotvec)
            x, y, z, w = ref.as_quat()
            q = Quaternion(w, x, y, z)
            euler = _v(q.toEulerAngles())
            np.testing.assert_allclose(euler, ref.as_euler("xyz"), atol=ATOL)

    def test_rotate_vector(self):
        for roll, pitch, yaw in _EULERS:
            q = Quaternion.fromEulerAngles(Vector3(roll, pitch, yaw))
            ref = _q_to_scipy(q)
            for v in self._SAMPLE_VECS:
                np.testing.assert_allclose(_v(q.rotate(Vector3(*v))), ref.apply(v), atol=ATOL)

    def test_multiplication_composition(self):
        q1 = Quaternion.fromEulerAngles(Vector3(0.3, -0.2, 0.5))
        q2 = Quaternion.fromEulerAngles(Vector3(-0.1, 0.4, 0.2))
        ref = _q_to_scipy(q1) * _q_to_scipy(q2)
        for v in self._SAMPLE_VECS:
            np.testing.assert_allclose(_v((q1 * q2).rotate(Vector3(*v))), ref.apply(v), atol=ATOL)

    def test_conjugate_is_inverse(self):
        for roll, pitch, yaw in _EULERS:
            q = Quaternion.fromEulerAngles(Vector3(roll, pitch, yaw))
            ref = _q_to_scipy(q).inv()
            for v in self._SAMPLE_VECS:
                np.testing.assert_allclose(_v(q.conjugate().rotate(Vector3(*v))), ref.apply(v), atol=ATOL)

    def test_from_axis_angle(self):
        axes = np.array([[1.0, 2.0, -1.0], [0.0, 0.0, 1.0], [-3.0, 1.0, 2.0]])
        for raw_axis in axes:
            axis = raw_axis / np.linalg.norm(raw_axis)
            for angle in (0.0, 0.7, 2.5, -1.3):
                q = Quaternion.fromAxisAngle(Vector3(*axis), angle)
                ref = Rotation.from_rotvec(axis * angle)
                for v in self._SAMPLE_VECS:
                    np.testing.assert_allclose(_v(q.rotate(Vector3(*v))), ref.apply(v), atol=ATOL)

    def test_from_rotation_matrix(self):
        for roll, pitch, yaw in _EULERS:
            ref = Rotation.from_euler("xyz", [roll, pitch, yaw])
            mat = ref.as_matrix()
            q = Quaternion.fromRotationMatrix(mat)
            for v in self._SAMPLE_VECS:
                np.testing.assert_allclose(_v(q.rotate(Vector3(*v))), ref.apply(v), atol=ATOL)

    def test_normalized_quaternion_unit_norm(self):
        # A normalized rotation quaternion should have unit norm like SciPy's.
        for roll, pitch, yaw in _EULERS:
            q = Quaternion.fromEulerAngles(Vector3(roll, pitch, yaw)).norm()
            self.assertAlmostEqual(abs(q), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
