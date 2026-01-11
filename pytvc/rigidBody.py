from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from loguru import logger
from typing import Iterable

@dataclass
class Vector3:

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        """Initialize a Vector3 object

        Args:
            x (float, optional): x component of the vector. Defaults to 0.0.
            y (float, optional): y component of the vector. Defaults to 0.0.
            z (float, optional): z component of the vector. Defaults to 0.0.
        """
        self.x: float | int = x
        self.y: float | int = y
        self.z: float | int = z

    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: float | Vector3) -> Vector3:
        if isinstance(other, Vector3):
            return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)
        return Vector3(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other: float | Vector3) -> Vector3:
        if isinstance(other, Vector3):
            return Vector3(self.x / other.x, self.y / other.y, self.z / other.z)
        return Vector3(self.x / other, self.y / other, self.z / other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector3): return False
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __iter__(self) -> Iterable[float]:
        return iter([self.x, self.y, self.z])

    def __getitem__(self, index: int) -> float:
        return [self.x, self.y, self.z][index]

    def __abs__(self) -> float:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def len(self) -> float:
        """Calculate the magnitude of a Vector3 object

        Returns:
            float: Magnitude of the Vector3 object
        """
        return abs(self)

    def norm(self) -> Vector3:
        """Normalize a Vector3 object

        Returns:
            Vector3: Normalized Vector3 object
        """
        try:
            ret: Vector3 = self / abs(self)
        except ZeroDivisionError:
            logger.warning("Vector3 norm: Division by zero")
            ret = Vector3(0.0, 0.0, 0.0)
        return ret

    def cross(self, other: Vector3) -> Vector3:
        """Calculate the cross product of two Vector3 objects

        Args:
            other (Vector3): Vector3 object to calculate the cross product with

        Returns:
            Vector3: Cross product of the two Vector3 objects
        """
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def dot(self, other: Vector3) -> float:
        """Calculate the dot product of two Vector3 objects

        Args:
            other (Vector3): Vector3 object to calculate the dot product with

        Returns:
            float: Dot product of the two Vector3 objects
        """
        return self.x * other.x + self.y * other.y + self.z * other.z

    def angleBetween(self, other: Vector3) -> float:
        """Calculate the angle between two Vector3 objects

        Args:
            other (Vector3): Vector3 object to calculate the angle with

        Returns:
            float: Angle between the two Vector3 objects in radians
        """
        try:
            inp: float = self.dot(other) / (abs(self) * abs(other))
            if abs(inp) > 1.0:
                logger.warning(
                    "Vector3 angleBetween: Value out of range, clamping to 1.0"
                )
                inp = np.clip(inp, -1.0, 1.0)
            ret: float = np.arccos(inp)
        except ZeroDivisionError:
            logger.warning("Vector3 angleBetween: Division by zero")
            ret: float = 0.0
        return ret

    def __repr__(self) -> str:
        return f"Vector3({self.x}, {self.y}, {self.z})"

    def __str__(self) -> str:
        return self.__repr__()


@dataclass
class Quaternion:

    def __init__(
        self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0
    ) -> None:
        """Initialize a Quaternion object

        Args:
            w (float, optional): Real component of the quaternion. Defaults to 1.0.
            x (float, optional): i component of the quaternion. Defaults to 0.0.
            y (float, optional): j component of the quaternion. Defaults to 0.0.
            z (float, optional): k component of the quaternion. Defaults to 0.0.
        """
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: Quaternion) -> Quaternion:
        return Quaternion(
            self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z
        )

    def __sub__(self, other: Quaternion) -> Quaternion:
        return Quaternion(
            self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z
        )

    def __mul__(self, other: Quaternion) -> Quaternion:
        if not isinstance(other, Quaternion):
            return Quaternion(
                self.w * other, self.x * other, self.y * other, self.z * other
            )
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def __truediv__(self, other: float | int) -> Quaternion:
        try:
            divisor = float(other)
            return Quaternion(
                self.w / divisor, self.x / divisor, self.y / divisor, self.z / divisor
            )
        except ZeroDivisionError:
            logger.warning("Quaternion division by zero")

            # Return a zero quaternion if division by zero occurs
            return Quaternion(0.0, 0.0, 0.0, 0.0)
        except TypeError:
            logger.warning(f"Quaternion division by non-numeric type {type(other)}")

            # Return a zero quaternion if division by non-float occurs
            return Quaternion(0.0, 0.0, 0.0, 0.0)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quaternion): return False
        return (
            self.w == other.w
            and self.x == other.x
            and self.y == other.y
            and self.z == other.z
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __iter__(self) -> Iterable[float]:
        return iter([self.w, self.x, self.y, self.z])

    def __getitem__(self, index: int) -> float:
        return [self.w, self.x, self.y, self.z][index]

    def conjugate(self) -> Quaternion:
        """Calculate the conjugate of a Quaternion object

        Returns:
            Quaternion: Conjugate of the Quaternion object
        """
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def __abs__(self) -> float:
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def len(self) -> float:
        """Calculate the magnitude of a Quaternion object

        Returns:
            float: Magnitude of the Quaternion object
        """
        return abs(self)

    def norm(self) -> Quaternion:
        """Normalize a Quaternion object

        Returns:
            Quaternion: Normalized Quaternion object
        """
        len = abs(self)
        if len == 0:
            logger.warning("Quaternion norm: Division by zero")
            return Quaternion(0.0, 0.0, 0.0, 0.0)
        return Quaternion(self.w / len, self.x / len, self.y / len, self.z / len)

    def rotate(self, v: Vector3) -> Vector3:
        """Rotate a Vector3 object by a Quaternion object

        Args:
            v (Vector3): Vector3 object to rotate

        Returns:
            Vector3: Rotated Vector3 object
        """
        try:
            qv = Quaternion(0, v.x, v.y, v.z)
        except TypeError:
            logger.warning("Quaternion rotate: Invalid Vector3 object")
            return Vector3(0.0, 0.0, 0.0)
        return (self * qv * self.conjugate()).xyz

    @property
    def xyz(self) -> Vector3:
        """Get the xyz components of a Quaternion object

        Returns:
            Vector3: xyz components of the Quaternion object
        """
        return Vector3(self.x, self.y, self.z)

    def dot(self, other: Quaternion) -> float:
        """Calculate the dot product of two Quaternion objects

        Args:
            other (Quaternion): Quaternion object to calculate the dot product with

        Returns:
            float: Dot product of the two Quaternion objects
        """
        return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z

    @staticmethod
    def fromAxisAngle(axis: Vector3, angle: float) -> Quaternion:
        """Create a Quaternion object from an axis and an angle

        Args:
            axis (Vector3): Axis of rotation
            angle (float): Angle of rotation

        Returns:
            Quaternion: Quaternion object representing the rotation
        """
        halfAngle = angle / 2
        return Quaternion(
            np.cos(halfAngle),
            axis.x * np.sin(halfAngle),
            axis.y * np.sin(halfAngle),
            axis.z * np.sin(halfAngle),
        )

    def toAxisAngle(self) -> tuple[Vector3, float]:
        """Convert a Quaternion object to an axis and an angle

        Returns:
            tuple[Vector3, float]: Axis and angle of rotation
        """
        angle = 2 * np.arccos(self.w)
        axis = self.xyz / np.sin(angle / 2)
        return axis, angle

    @staticmethod
    def fromEulerAngles(rot: Vector3) -> Quaternion:
        """Create a Quaternion object from Euler angles

        Args:
            rot (Vector3): Euler angles

        Returns:
            Quaternion: Quaternion object representing the rotation
        """
        cy = np.cos(rot.z * 0.5)
        sy = np.sin(rot.z * 0.5)
        cp = np.cos(rot.y * 0.5)
        sp = np.sin(rot.y * 0.5)
        cr = np.cos(rot.x * 0.5)
        sr = np.sin(rot.x * 0.5)

        return Quaternion(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )

    @staticmethod
    def fromRotationMatrix(mat) -> Quaternion:
        """Create a Quaternion from a 3x3 rotation matrix.

        Args:
            mat: Rotation matrix as a list of lists or numpy array. A 4x4
                homogeneous matrix is also accepted; only the upper-left 3x3
                block is used.

        Returns:
            Quaternion: Quaternion representing the same rotation.
        """
        m = np.asarray(mat, dtype=float)
        if m.shape == (4, 4):
            m = m[:3, :3]
        if m.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3 or 4x4")

        trace = np.trace(m)
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s

        return Quaternion(w, x, y, z).norm()

    def toEulerAngles(self) -> Vector3:
        """Convert a Quaternion object to Euler angles

        Returns:
            Vector3: Vector3 object containing roll, pitch, and yaw angles
        """
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x**2 + self.y**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (self.w * self.y - self.z * self.x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y**2 + self.z**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return Vector3(roll, pitch, yaw)

    def __repr__(self) -> str:
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"

    def __str__(self) -> str:
        return self.__repr__()


class RigidBody:

    def __init__(
        self,
        mass: float,
        inertia: Vector3,
        position: Vector3,
        velocity: Vector3,
        rotation: Quaternion,
        rotVel: Vector3,
    ) -> None:
        """Initialize a RigidBody object

        Args:
            mass (float): Mass of the rigid body
            inertia (Vector3): Moment of inertia of the rigid body
            position (Vector3): Position of the rigid body
            velocity (Vector3): Velocity of the rigid body
            rotation (Quaternion): Rotation of the rigid body
            rotVel (Vector3): Angular velocity of the rigid body
        """
        self.mass = mass
        self.inertia = inertia
        self.position = position
        self.velocity = velocity
        self.rotation = rotation
        self.rotVel = rotVel

        self._torque = Vector3()
        self._accel = Vector3()
        self._lastAccel = Vector3()

    def getAccel(self) -> Vector3:
        """Get the acceleration of the rigid body

        Returns:
            Vector3: Acceleration of the rigid body
        """
        return self._lastAccel

    def applyTorque(self, torque: Vector3) -> None:
        """Apply a torque to the rigid body

        Args:
            torque (Vector3): Torque to apply
        """
        self._torque += torque / self.inertia
    
    def applyLocalTorque(self, torque: Vector3) -> None:
        """Apply a local torque to the rigid body

        Args:
            torque (Vector3): Torque to apply
        """
        self._torque += self.rotation.rotate(torque) / self.inertia

    def applyForce(self, force: Vector3, position: Vector3) -> None:
        """Apply a force to the rigid body

        Args:
            force (Vector3): Force to apply
        """
        self._accel += force / self.mass
        self.applyTorque(position.cross(force))

    def applyLocalForce(self, force: Vector3, position: Vector3) -> None:
        """Apply a local force to the rigid body

        Args:
            force (Vector3): Force to apply
        """
        self._accel += self.rotation.rotate(force) / self.mass
        self.applyTorque(position.cross(force))

    def update(self, dt: float) -> None:
        """Update the rigid body

        Args:
            dt (float): Time step
        """
        self.velocity += self._accel * dt
        self.position += self.velocity * dt

        rotVelMag = abs(self.rotVel)
        if abs(rotVelMag) > 0:
            axis = self.rotVel.norm()
            self.rotation: Quaternion = (
                Quaternion.fromAxisAngle(axis, rotVelMag * dt) * self.rotation
            )
            self.rotation = self.rotation.norm()

        self.rotVel += self._torque * dt

        self._lastAccel = self._accel

        self._torque = Vector3()
        self._accel = Vector3()
