from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from loguru import logger

try:
    from numba import njit, float64, types  # type: ignore
    NUMBA_ENABLED = True

    _ROTATE_SIG = types.UniTuple(float64, 3)(float64, float64, float64, float64, float64, float64, float64)
    _QMUL_SIG = types.UniTuple(float64, 4)(float64, float64, float64, float64, float64, float64, float64, float64)
    _NORMQ_SIG = types.UniTuple(float64, 4)(float64, float64, float64, float64)
    _AXANG_SIG = types.UniTuple(float64, 4)(float64, float64, float64, float64)
except Exception:  # pragma: no cover - fallback when numba is unavailable
    NUMBA_ENABLED = False
    float64 = None
    types = None

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    _ROTATE_SIG = None
    _QMUL_SIG = None
    _NORMQ_SIG = None
    _AXANG_SIG = None  

@njit(_ROTATE_SIG, cache=True, fastmath=True)
def _rotate_xyz_kernel(
    qw: float,
    qx: float,
    qy: float,
    qz: float,
    vx: float,
    vy: float,
    vz: float,
) -> tuple[float, float, float]:
    """Rotate vector (vx, vy, vz) by unit quaternion (qw, qx, qy, qz)."""
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)

    rx = vx + qw * tx + (qy * tz - qz * ty)
    ry = vy + qw * ty + (qz * tx - qx * tz)
    rz = vz + qw * tz + (qx * ty - qy * tx)
    return rx, ry, rz


@njit(_QMUL_SIG, cache=True, fastmath=True)
def _quat_mul_kernel(
    aw: float,
    ax: float,
    ay: float,
    az: float,
    bw: float,
    bx: float,
    by: float,
    bz: float,
) -> tuple[float, float, float, float]:
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


@njit(_NORMQ_SIG, cache=True, fastmath=True)
def _normalize_quat_kernel(
    w: float,
    x: float,
    y: float,
    z: float,
) -> tuple[float, float, float, float]:
    mag2 = w * w + x * x + y * y + z * z
    if mag2 == 0.0:
        return 1.0, 0.0, 0.0, 0.0
    inv = 1.0 / math.sqrt(mag2)
    return w * inv, x * inv, y * inv, z * inv


@njit(_AXANG_SIG, cache=True, fastmath=True)
def _axis_angle_unit_quat_kernel(
    axis_x: float,
    axis_y: float,
    axis_z: float,
    angle: float,
) -> tuple[float, float, float, float]:
    half = 0.5 * angle
    s = math.sin(half)
    c = math.cos(half)
    return c, axis_x * s, axis_y * s, axis_z * s




def warm_numba_cache() -> None:
    """Force compilation of the hot-path kernels once for float64 signatures.

    Useful if you want to pay the JIT cost during startup instead of on the
    first simulation step. Safe to call multiple times.
    """
    if not NUMBA_ENABLED:
        return
    _rotate_xyz_kernel(1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0)
    _quat_mul_kernel(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    _normalize_quat_kernel(1.0, 0.0, 0.0, 0.0)
    _axis_angle_unit_quat_kernel(1.0, 0.0, 0.0, 0.1)

# Automatically warm cache
warm_numba_cache()

class Vector3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z

    def copy(self) -> Vector3:
        return Vector3(self.x, self.y, self.z)

    def set(self, x: float, y: float, z: float) -> Vector3:
        self.x = x
        self.y = y
        self.z = z
        return self

    def zero(self) -> Vector3:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        return self

    def copy_from(self, other: Vector3) -> Vector3:
        self.x = other.x
        self.y = other.y
        self.z = other.z
        return self

    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: float | Vector3) -> Vector3:
        if isinstance(other, Vector3):
            return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)
        return Vector3(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other: float) -> Vector3:
        return Vector3(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other: float | Vector3) -> Vector3:
        if isinstance(other, Vector3):
            return Vector3(self.x / other.x, self.y / other.y, self.z / other.z)
        return Vector3(self.x / other, self.y / other, self.z / other)

    def __iadd__(self, other: Vector3) -> Vector3:
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self

    def __isub__(self, other: Vector3) -> Vector3:
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z
        return self

    def add_scaled(self, other: Vector3, scale: float) -> Vector3:
        self.x += other.x * scale
        self.y += other.y * scale
        self.z += other.z * scale
        return self

    def add_div_components(self, a: Vector3, b: Vector3) -> Vector3:
        self.x += a.x / b.x
        self.y += a.y / b.y
        self.z += a.z / b.z
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector3):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __iter__(self) -> Iterable[float]:
        yield self.x
        yield self.y
        yield self.z

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.x
        if index == 1:
            return self.y
        if index == 2:
            return self.z
        raise IndexError(index)

    def mag2(self) -> float:
        x = self.x
        y = self.y
        z = self.z
        return x * x + y * y + z * z

    def __abs__(self) -> float:
        return math.sqrt(self.mag2())

    def len(self) -> float:
        return abs(self)

    def norm(self) -> Vector3:
        mag2 = self.mag2()
        if mag2 == 0.0:
            logger.warning("Vector3 norm: Division by zero")
            return Vector3(0.0, 0.0, 0.0)
        inv = 1.0 / math.sqrt(mag2)
        return Vector3(self.x * inv, self.y * inv, self.z * inv)

    def cross(self, other: Vector3) -> Vector3:
        ax = self.x
        ay = self.y
        az = self.z
        bx = other.x
        by = other.y
        bz = other.z
        return Vector3(
            ay * bz - az * by,
            az * bx - ax * bz,
            ax * by - ay * bx,
        )

    def dot(self, other: Vector3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def angleBetween(self, other: Vector3) -> float:
        a2 = self.mag2()
        b2 = other.mag2()
        if a2 == 0.0 or b2 == 0.0:
            logger.warning("Vector3 angleBetween: Division by zero")
            return 0.0
        inp = self.dot(other) / math.sqrt(a2 * b2)
        if inp < -1.0:
            inp = -1.0
        elif inp > 1.0:
            inp = 1.0
        return math.acos(inp)

    def __repr__(self) -> str:
        return f"Vector3({self.x}, {self.y}, {self.z})"

    def __str__(self) -> str:
        return self.__repr__()


class Quaternion:
    __slots__ = ("w", "x", "y", "z")

    def __init__(
        self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0
    ) -> None:
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def copy(self) -> Quaternion:
        return Quaternion(self.w, self.x, self.y, self.z)

    def set(self, w: float, x: float, y: float, z: float) -> Quaternion:
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        return self

    def __add__(self, other: Quaternion) -> Quaternion:
        return Quaternion(
            self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z
        )

    def __sub__(self, other: Quaternion) -> Quaternion:
        return Quaternion(
            self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z
        )

    def __mul__(self, other: float | Quaternion) -> Quaternion:
        if not isinstance(other, Quaternion):
            return Quaternion(
                self.w * other, self.x * other, self.y * other, self.z * other
            )
        w, x, y, z = _quat_mul_kernel(
            self.w, self.x, self.y, self.z, other.w, other.x, other.y, other.z
        )
        return Quaternion(w, x, y, z)

    def __rmul__(self, other: float) -> Quaternion:
        return Quaternion(
            self.w * other, self.x * other, self.y * other, self.z * other
        )

    def __truediv__(self, other: float | int) -> Quaternion:
        divisor = float(other)
        if divisor == 0.0:
            logger.warning("Quaternion division by zero")
            return Quaternion(0.0, 0.0, 0.0, 0.0)
        return Quaternion(
            self.w / divisor, self.x / divisor, self.y / divisor, self.z / divisor
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quaternion):
            return False
        return (
            self.w == other.w
            and self.x == other.x
            and self.y == other.y
            and self.z == other.z
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __iter__(self) -> Iterable[float]:
        yield self.w
        yield self.x
        yield self.y
        yield self.z

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.w
        if index == 1:
            return self.x
        if index == 2:
            return self.y
        if index == 3:
            return self.z
        raise IndexError(index)

    def conjugate(self) -> Quaternion:
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def mag2(self) -> float:
        w = self.w
        x = self.x
        y = self.y
        z = self.z
        return w * w + x * x + y * y + z * z

    def __abs__(self) -> float:
        return math.sqrt(self.mag2())

    def len(self) -> float:
        return abs(self)

    def norm(self) -> Quaternion:
        w, x, y, z = _normalize_quat_kernel(self.w, self.x, self.y, self.z)
        if w == 1.0 and x == 0.0 and y == 0.0 and z == 0.0 and self.mag2() == 0.0:
            logger.warning("Quaternion norm: Division by zero")
        return Quaternion(w, x, y, z)

    def normalize_ip(self) -> Quaternion:
        old_mag2 = self.mag2()
        self.w, self.x, self.y, self.z = _normalize_quat_kernel(
            self.w, self.x, self.y, self.z
        )
        if old_mag2 == 0.0:
            logger.warning("Quaternion normalize_ip: Division by zero")
        return self

    def rotate_into(self, v: Vector3, out: Vector3) -> Vector3:
        """Rotate a vector by this quaternion into a preallocated output vector.

        This hot-path method assumes the quaternion is already normalized.
        """
        out.x, out.y, out.z = _rotate_xyz_kernel(
            self.w, self.x, self.y, self.z, v.x, v.y, v.z
        )
        return out

    def rotate_xyz_into(self, vx: float, vy: float, vz: float, out: Vector3) -> Vector3:
        """Rotate raw xyz scalars into a preallocated output vector.

        This hot-path method assumes the quaternion is already normalized.
        """
        out.x, out.y, out.z = _rotate_xyz_kernel(
            self.w, self.x, self.y, self.z, vx, vy, vz
        )
        return out

    def rotate(self, v: Vector3) -> Vector3:
        """Rotate a Vector3 by this quaternion.

        This method assumes the quaternion is already normalized. Use rotateSafe()
        if you need the old normalize-on-every-call behavior.
        """
        out = Vector3()
        return self.rotate_into(v, out)

    def rotateSafe(self, v: Vector3) -> Vector3:
        """Compatibility path that normalizes first."""
        q = self.norm()
        out = Vector3()
        return q.rotate_into(v, out)

    @property
    def xyz(self) -> Vector3:
        return Vector3(self.x, self.y, self.z)

    def dot(self, other: Quaternion) -> float:
        return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z

    @staticmethod
    def fromAxisAngle(axis: Vector3, angle: float) -> Quaternion:
        w, x, y, z = _axis_angle_unit_quat_kernel(axis.x, axis.y, axis.z, angle)
        return Quaternion(w, x, y, z)

    def toAxisAngle(self) -> tuple[Vector3, float]:
        q = self.norm()
        w = max(-1.0, min(1.0, q.w))
        angle = 2.0 * math.acos(w)
        s = math.sin(0.5 * angle)
        if s == 0.0:
            return Vector3(1.0, 0.0, 0.0), 0.0
        return Vector3(q.x / s, q.y / s, q.z / s), angle

    @staticmethod
    def fromEulerAngles(rot: Vector3) -> Quaternion:
        hx = 0.5 * rot.x
        hy = 0.5 * rot.y
        hz = 0.5 * rot.z
        cr = math.cos(hx)
        sr = math.sin(hx)
        cp = math.cos(hy)
        sp = math.sin(hy)
        cy = math.cos(hz)
        sy = math.sin(hz)
        return Quaternion(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )

    @staticmethod
    def fromRotationMatrix(mat) -> Quaternion:
        m = np.asarray(mat, dtype=float)
        if m.shape == (4, 4):
            m = m[:3, :3]
        if m.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3 or 4x4")

        trace = float(np.trace(m))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s

        q = Quaternion(w, x, y, z)
        q.normalize_ip()
        return q

    def toEulerAngles(self) -> Vector3:
        q = self.norm()
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return Vector3(roll, pitch, yaw)

    def __repr__(self) -> str:
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"

    def __str__(self) -> str:
        return self.__repr__()


class RigidBody:
    __slots__ = (
        "mass",
        "inv_mass",
        "inertia",
        "position",
        "velocity",
        "rotation",
        "rotVel",
        "_torque",
        "_accel",
        "_lastAccel",
        "_tmp1",
        "_tmp2",
    )

    def __init__(
        self,
        mass: float,
        inertia: Vector3,
        position: Vector3,
        velocity: Vector3,
        rotation: Quaternion,
        rotVel: Vector3,
    ) -> None:
        self.mass = mass
        self.inv_mass = 1.0 / mass
        self.inertia = inertia
        self.position = position
        self.velocity = velocity
        self.rotation = rotation.norm()
        self.rotVel = rotVel

        self._torque = Vector3()
        self._accel = Vector3()
        self._lastAccel = Vector3()
        self._tmp1 = Vector3()
        self._tmp2 = Vector3()

    def getAccel(self) -> Vector3:
        return self._lastAccel

    def applyTorque(self, torque: Vector3) -> None:
        self._torque.x += torque.x / self.inertia.x
        self._torque.y += torque.y / self.inertia.y
        self._torque.z += torque.z / self.inertia.z

    def applyLocalTorque(self, torque: Vector3) -> None:
        tmp = self._tmp1
        self.rotation.rotate_into(torque, tmp)
        self._torque.x += tmp.x / self.inertia.x
        self._torque.y += tmp.y / self.inertia.y
        self._torque.z += tmp.z / self.inertia.z

    def applyForce(self, force: Vector3, position: Vector3) -> None:
        self._accel.x += force.x * self.inv_mass
        self._accel.y += force.y * self.inv_mass
        self._accel.z += force.z * self.inv_mass

        px = position.x
        py = position.y
        pz = position.z
        fx = force.x
        fy = force.y
        fz = force.z

        tx = py * fz - pz * fy
        ty = pz * fx - px * fz
        tz = px * fy - py * fx

        self._torque.x += tx / self.inertia.x
        self._torque.y += ty / self.inertia.y
        self._torque.z += tz / self.inertia.z

    def applyLocalForce(self, force: Vector3, position: Vector3) -> None:
        world_force = self._tmp1
        self.rotation.rotate_into(force, world_force)

        self._accel.x += world_force.x * self.inv_mass
        self._accel.y += world_force.y * self.inv_mass
        self._accel.z += world_force.z * self.inv_mass

        px = position.x
        py = position.y
        pz = position.z
        fx = force.x
        fy = force.y
        fz = force.z

        tx = py * fz - pz * fy
        ty = pz * fx - px * fz
        tz = px * fy - py * fx

        world_torque = self._tmp2
        self.rotation.rotate_xyz_into(tx, ty, tz, world_torque)

        self._torque.x += world_torque.x / self.inertia.x
        self._torque.y += world_torque.y / self.inertia.y
        self._torque.z += world_torque.z / self.inertia.z

    def update(self, dt: float) -> None:
        self.velocity.x += self._accel.x * dt
        self.velocity.y += self._accel.y * dt
        self.velocity.z += self._accel.z * dt

        self.position.x += self.velocity.x * dt
        self.position.y += self.velocity.y * dt
        self.position.z += self.velocity.z * dt

        wx = self.rotVel.x
        wy = self.rotVel.y
        wz = self.rotVel.z
        wmag2 = wx * wx + wy * wy + wz * wz

        if wmag2 > 0.0:
            wmag = math.sqrt(wmag2)
            inv = 1.0 / wmag
            dw, dx, dy, dz = _axis_angle_unit_quat_kernel(
                wx * inv, wy * inv, wz * inv, wmag * dt
            )
            rw, rx, ry, rz = _quat_mul_kernel(
                dw, dx, dy, dz,
                self.rotation.w, self.rotation.x, self.rotation.y, self.rotation.z,
            )
            self.rotation.w, self.rotation.x, self.rotation.y, self.rotation.z = _normalize_quat_kernel(
                rw, rx, ry, rz
            )

        self.rotVel.x += self._torque.x * dt
        self.rotVel.y += self._torque.y * dt
        self.rotVel.z += self._torque.z * dt

        self._lastAccel.copy_from(self._accel)
        self._torque.zero()
        self._accel.zero()
