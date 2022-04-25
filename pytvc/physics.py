from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy as sp

def clamp(x, min_val, max_val):
    return max(min(x, max_val), min_val)

@dataclass
class vec3:

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        """__init__ initializes a vec3 object.

        Args:
            x (float, optional): the x component of the vector. Defaults to 0.0.
            y (float, optional): the y component of the vector. Defaults to 0.0.
            z (float, optional): the z component of the vector. Defaults to 0.0.
        """
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: vec3) -> vec3:
        """__add__ adds 2 vectors and returns the sum

        Args:
            other (vec3): right hand vector to add

        Raises:
            NotImplementedError: raises notImplemented if the right hand side is not a vector object

        Returns:
            vec3: a vector sum of the left hand and right hand vector
        """
        if isinstance(other, vec3):
            return vec3(self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            raise NotImplementedError

    def __sub__(self, other: vec3) -> vec3:
        """__sub__ subtracts 2 vectors and returns the difference

        Args:
            other (vec3): right hand vector to subtract

        Raises:
            NotImplementedError: raises notImplemented if the right hand side is not a vector object

        Returns:
            vec3: a vector difference of the left hand and right hand vector
        """
        if isinstance(other, vec3):
            return vec3(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            raise NotImplementedError

    def __mul__(self, other: vec3) -> vec3:
        """__mul__ multiplies 2 vectors and returns the product

        Args:
            other (vec3): right hand vector to multiply

        Raises:
            NotImplementedError: raises notImplemented if the right hand side is not a vector object

        Returns:
            vec3: a vector product of the left hand and right hand vector
        """
        if isinstance(other, vec3):
            return vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        elif isinstance(other, int) or isinstance(other, float):
            return vec3(self.x * other, self.y * other, self.z * other)
        else:
            raise NotImplementedError

    def __truediv__(self, other) -> vec3:
        """__truediv__ divides 2 vectors and returns the quotient

        Args:
            other: right hand vector to divide or scalar to divide by

        Raises:
            NotImplementedError: raises notImplemented if the right hand side is not a vector object or a number

        Returns:
            vec3: a vector quotient of the left hand and right hand vector
        """
        if isinstance(other, vec3):
            return vec3(self.x / other.x, self.y / other.y, self.z / other.z)
        elif isinstance(other, int) or isinstance(other, float):
            return vec3(self.x / other, self.y / other, self.z / other)
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        """__repr__ returns a string representation of the vector

        Returns:
            str: a string representation of the vector
        """
        return f"vec3({self.x}, {self.y}, {self.z})"
    
    def __str__(self) -> str:
        """__str__ returns a string representation of the vector

        Returns:
            str: a string representation of the vector
        """
        return f"{self.x}, {self.y}, {self.z}"
    
    def __neg__(self) -> vec3:
        """__neg__ returns the negative of the vector

        Returns:
            vec3: the negative of the vector
        """
        return vec3(-self.x, -self.y, -self.z)
    
    def __eq__(self, other: vec3) -> bool:
        """__eq__ returns true if the vectors are equal

        Args:
            other (vec3): right hand vector to compare

        Raises:
            NotImplementedError: raises notImplemented if the right hand side is not a vector object

        Returns:
            bool: true if the vectors are equal
        """
        if isinstance(other, vec3):
            return self.x == other.x and self.y == other.y and self.z == other.z
        else:
            raise NotImplementedError
    
    def __ne__(self, other: vec3) -> bool:
        """__ne__ returns true if the vectors are not equal

        Args:
            other (vec3): right hand vector to compare

        Raises:
            NotImplementedError: raises notImplemented if the right hand side is not a vector object

        Returns:
            bool: true if the vectors are not equal
        """
        if isinstance(other, vec3):
            return self.x != other.x or self.y != other.y or self.z != other.z
        else:
            raise NotImplementedError

    def __round__(self, ndigits: int = 0) -> vec3:
        """__round__ returns the rounded vector

        Args:
            ndigits (int): number of digits to round to

        Returns:
            vec3: the rounded vector
        """
        return vec3(round(self.x, ndigits), round(self.y, ndigits), round(self.z, ndigits))
    
    def __abs__(self) -> vec3:
        """__abs__ returns the absolute value of the vector

        Returns:
            vec3: the absolute value of the vector
        """
        return vec3(abs(self.x), abs(self.y), abs(self.z))
    
    def cross(self, other: vec3) -> vec3:
        """cross returns the cross product of 2 vectors

        Args:
            other (vec3): right hand vector to cross

        Raises:
            NotImplementedError: raises notImplemented if the right hand side is not a vector object

        Returns:
            vec3: a vector cross product of the left hand and right hand vector
        """
        if isinstance(other, vec3):
            return vec3(self.y * other.z - self.z * other.y, self.z * other.x - self.x * other.z, self.x * other.y - self.y * other.x)
        else:
            raise NotImplementedError
    
    def dot(self, other: vec3) -> float:
        """dot returns the dot product of 2 vectors

        Args:
            other (vec3): right hand vector to dot

        Raises:
            NotImplementedError: raises notImplemented if the right hand side is not a vector object

        Returns:
            float: a vector dot product of the left hand and right hand vector
        """
        if isinstance(other, vec3):
            return self.x * other.x + self.y * other.y + self.z * other.z
        else:
            raise NotImplementedError
    
    def length(self) -> float:
        """length returns the length of the vector

        Returns:
            float: the length of the vector
        """
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> vec3:
        """normalize returns the normalized vector

        Returns:
            vec3: the normalized vector
        """
        return self / self.length()

    def angle_between_vectors(self, vector: vec3) -> float:
        """Calculate the angle between two vectors."""
        if isinstance(vector, vec3):
            return np.arccos(self.dot(vector) / (self.length() * vector.length()))
        else:
            return None

@dataclass
class quaternion:

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """__init__ initializes a quaternion object

        Args:
            w (float, optional): the real component of the quaternion. Defaults to 1.0.
            x (float, optional): the x component of the quaternion. Defaults to 0.0.
            y (float, optional): the y component of the quaternion. Defaults to 0.0.
            z (float, optional): the z component of the quaternion. Defaults to 0.0.
        """
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __mul__(self, other: quaternion) -> quaternion:
        """__mul__ multiplies 2 quaternions and returns the product

        Args:
            other (quaternion): right hand quaternion to multiply

        Raises:
            NotImplementedError: raises notImplemented if the right hand side is not a quaternion object

        Returns:
            quaternion: a quaternion product of the left hand and right hand quaternion
        """
        if isinstance(other, quaternion):
            return quaternion(
                self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
                self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
                self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
                self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            )
        else:
            raise NotImplementedError
    
    def __add__(self, other: quaternion) -> quaternion:
        """__add__ adds 2 quaternions and returns the sum

        Args:
            other (quaternion): right hand quaternion to add

        Raises:
            NotImplementedError: raises notImplemented if the right hand side is not a quaternion object

        Returns:
            quaternion: a quaternion sum of the left hand and right hand quaternion
        """
        if isinstance(other, quaternion):
            return quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            raise NotImplementedError

    def __sub__(self, other: quaternion) -> quaternion:
        """__sub__ subtracts 2 quaternions and returns the difference

        Args:
            other (quaternion): right hand quaternion to subtract

        Raises:
            NotImplementedError: raises notImplemented if the right hand side is not a quaternion object

        Returns:
            quaternion: a quaternion difference of the left hand and right hand quaternion
        """
        if isinstance(other, quaternion):
            return quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        """__repr__ returns the quaternion as a string

        Returns:
            str: the quaternion as a string
        """
        return f"quaternion({self.w}, {self.x}, {self.y}, {self.z})"
    
    def __str__(self) -> str:
        """__str__ returns the quaternion as a string

        Returns:
            str: the quaternion as a string
        """
        return f"quaternion({self.w}, {self.x}, {self.y}, {self.z})"
    
    def length(self) -> float:
        """length returns the length of the quaternion

        Returns:
            float: the length of the quaternion
        """
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> quaternion:
        """normalize returns the normalized quaternion

        Returns:
            quaternion: the normalized quaternion
        """
        return self / self.length()

    def conjugate(self) -> quaternion:
        """conjugate returns the conjugate of the quaternion

        Returns:
            quaternion: the conjugate of the quaternion
        """
        return quaternion(self.w, -self.x, -self.y, -self.z)
    
    def rotate(self, vector: vec3) -> vec3:
        """rotate returns the rotated vector

        Args:
            vector (vec3): vector to rotate

        Raises:
            NotImplementedError: raises notImplemented if the right hand side is not a vector object

        Returns:
            vec3: the rotated vector
        """
        if isinstance(vector, vec3):
            tmp: quaternion = quaternion(0, vector.x, vector.y, vector.z)
            tmp = self * tmp * self.conjugate()
            return vec3(tmp.x, tmp.y, tmp.z)
        else:
            raise NotImplementedError

    def fractional(self, a) -> quaternion:
        """Return the fractional of the quaternion."""

        self.w = 1-a + a*self.w
        self.x *= a
        self.y *= a
        self.z *= a

        return self.norm()

    def rotation_between_vectors(self, vector: vec3) -> quaternion:
        """Return the rotation quaternion between two vectors.

        parameters:

        vector : vector3
            vector to rotate
        """

        q = quaternion(0.0, vector.x, vector.y, vector.z)

        q = self * q
        q.w = 1 - q.w

        return q.normalize()

    def from_axis_angle(self, axis: vec3, angle: float) -> quaternion:
        """Return the quaternion from an axis and angle.

        parameters:

        axis : vector3
            axis of rotation

        angle : float
            angle of rotation
        """

        sa: float = np.sin(angle / 2)

        self.w = np.cos(angle / 2)
        self.x = axis.x * sa
        self.y = axis.y * sa
        self.z = axis.z * sa

        return self

    def from_euler(self, euler_angles: vec3) -> quaternion:
        """set the quaternion from euler angles

        Args:
            euler_angles (vec3): euler angles to set to

        Raises:
            NotImplementedError: raises notImplemented if the right hand side is not a vector object

        Returns:
            quaternion: the quaternion set from euler angles
        """

        cr = np.cos(euler_angles.x / 2.0)
        cp = np.cos(euler_angles.y / 2.0)
        cy = np.cos(euler_angles.z / 2.0)

        sr = np.sin(euler_angles.x / 2.0)
        sp = np.sin(euler_angles.y / 2.0)
        sy = np.sin(euler_angles.z / 2.0)

        self.w = cr * cp * cy + sr * sp * sy
        self.x = sr * cp * cy - cr * sp * sy
        self.y = cr * sp * cy + sr * cp * sy
        self.z = cr * cp * sy - sr * sp * cy

        return self

    def to_euler(self) -> vec3:
        """Convert a quaternion to euler angles."""
        x = np.arctan2(2.0 * (self.w * self.x + self.y * self.z),
                       1.0 - 2.0 * (self.x**2 + self.y**2))
        y = np.arcsin(2.0 * (self.w * self.y - self.z * self.x))
        z = np.arctan2(2.0 * (self.w * self.z + self.x * self.y),
                       1.0 - 2.0 * (self.y**2 + self.z**2))

        return vec3(x, y, z)

class TVC:

    def __init__(self, speed: float = 0.0, offset: vec3 = vec3, throttleSpeed: float = 0.0, maxPosition: vec3 = vec3, minPosition: vec3 = vec3, minThrottle: float = 0.9, actuator_precision: float = 0.0, linkage_ratio: float = 0.0) -> None:
        """__init__ initializes the TVC object

        Args:
            speed (float, optional): speed of the servos. Defaults to 0.0.
            offset (vec3, optional): offset of the TVC mount. Defaults to vec3.
            throttleSpeed (float, optional): speed at which the throttling mechanism throttles. Defaults to 0.0.
            maxPosition (vec3, optional): maximum positive position of the mount. Defaults to vec3.
            minPosition (vec3, optional): maximum negative position of the mount. Defaults to vec3.
            minThrottle (float, optional): minimum throttle value. Defaults to 0.9.
            actuator_precision (float, optional): precision of the servos, if set to zero it will be infinite. Defaults to 0.0.
            linkage_ratio (float, optional): ratio between servo movement and TVC movement, if set to zero will skip calculation. Defaults to 0.0.
        """
        self._rotation: quaternion = quaternion()
        self._rotation_eulers: vec3 = vec3()
        self._throttle: float = 1.0

        self.offset: vec3 = offset
        self.maxPosition: vec3 = maxPosition
        self.minPosition: vec3 = minPosition
        self.targetEulers: vec3 = vec3()
        self.speed: float = speed
        self.offset: vec3 = vec3()
        self.targetThrottle: float = 1.0
        self.throttleSpeed: float = throttleSpeed
        self.minThrottle: float = minThrottle

    def throttle(self, target_throttle) -> None:
        self.target_throttle = target_throttle
    
    def set_position(self, target_position) -> None:
        if isinstance(target_position, quaternion):
            self.target_eulers = target_position.to_euler()
        elif isinstance(target_position, vec3):
            self.target_eulers = target_position
        else:
            raise NotImplemented
    
    def update(self, dt: float):
        """update updates the TVC mount and throttling mechanism

        Args:
            dt (float): time step
        """
        error: vec3 = self.target_eulers - self._rotation_eulers
        error = error * (self.speed * dt)
        self._rotation_eulers += error
        
        self._rotation_eulers.y = clamp(self._rotation_eulers.y, self.minPosition.y, self.maxPosition.y)
        self._rotation_eulers.z = clamp(self._rotation_eulers.z, self.minPosition.z, self.maxPosition.z)

        self._rotation = quaternion().from_euler(self._rotation_eulers + self.offset)

        throttle_error = self.target_throttle - self._throttle
        throttle_error = throttle_error * self.throttle_speed * dt
        self._throttle += throttle_error

        if self._throttle < self.minThrottle:
            self._throttle = self.minThrottle
        if self._throttle > 1.0:
            self._throttle = 1.0

    def calculate_forces(self, thrust: float) -> vec3:
        force: vec3 = self._rotation.rotate(vec3(thrust * self._throttle, 0.0, 0.0))
        return force
        

@dataclass
class physics_body:

    def __init__(self):
        """__init__ initializes the physics body object
        NEEDS SIGNIFICANT WORK"""
        self.position: vec3 = vec3(0.0, 0.0, 0.0)
        self.velocity: vec3 = vec3(0.0, 0.0, 0.0)

        self.acceleration: vec3 = vec3(0.0, 0.0, 0.0)
        self.acceleration_local: vec3 = vec3(0.0, 0.0, 0.0)

        self.rotation: quaternion = quaternion(1.0, 0.0, 0.0, 0.0)
        self.rotation_euler: vec3 = vec3(0.0, 0.0, 0.0)

        self.rotational_velocity: vec3 = vec3(0.0, 0.0, 0.0)
        self.rotational_acceleration: vec3 = vec3(0.0, 0.0, 0.0)

        self.mass: float = 1.0
        self.moment_of_inertia: vec3 = vec3(1.0, 1.0, 1.0)

        self.aoa: float = 0.0
        self.ref_area: float = 1.0

        self.drag_coefficient_forwards: float = 0.0
        self.drag_coefficient_sideways: float = 0.0

        self.wind: vec3 = vec3(0.0, 0.0, 0.0)

        self.cp_location: vec3 = vec3(0.5, 0.0, 0.0)

        self.friction_coeff: float = 0.0
        self.air_density: float = 1.225

        self.drag_force: vec3 = vec3()

        self.use_aero: bool = False

    def apply_torque(self, torque: vec3):
        """Applies torque in the global frame"""
        self.rotational_acceleration += (torque / self.moment_of_inertia)

    def apply_point_torqe(self, torque: vec3, point: vec3):
        """Applies point torque in the global frame"""
        tmp = point.cross(torque)
        self.apply_torque(tmp)

    def apply_local_torque(self, torque: vec3):
        """Applies torque in the local frame"""
        self.apply_torque(self.rotation.rotate(torque))

    def apply_local_point_torqe(self, torque: vec3, point: vec3):
        """Applies point torque in the local frame"""
        self.apply_point_torqe(self.rotation.rotate(
            torque), self.rotation.rotate(point))

    def apply_force(self, force: vec3):
        """Applies force in global frame"""
        accel: vec3 = force / self.mass
        self.acceleration += accel

    def apply_point_force(self, force: vec3, point: vec3):
        """Applies point force in global frame"""
        self.apply_force(force)
        self.apply_point_torqe(force, point)

    def apply_local_force(self, force: vec3):
        """Applies force in local frame"""
        self.apply_force(self.rotation.rotate(force))
    
    def apply_local_point_force(self, force: vec3, point: vec3):
        """Applies point force in local frame"""
        self.apply_local_force(self.rotation.rotate(force))
        self.apply_local_point_torqe(force, point)

    def update(self, dt: float):
        """Updates the physics body."""

        # drag

        velocity_relative_wind: vec3 = self.velocity - self.wind
        
        if velocity_relative_wind.length() > 0.0:

            self.aoa = velocity_relative_wind.angle_between_vectors(
                self.rotation.rotate(vec3(1.0, 0.0, 0.0)))

            if self.aoa <= 1e-5:
                self.aoa = 1e-5

            if (self.aoa > 1.5708):
               self.aoa = np.pi - self.aoa

            # self.aoa = local_course.angle_between_vectors(vec3(1.0, 0.0, 0.0))

            drag_coefficient = self.drag_coefficient_forwards + ((-np.cos(self.aoa)/2.0 + 0.5) * (self.drag_coefficient_sideways - self.drag_coefficient_forwards))

            self.drag_force = -velocity_relative_wind.normalize() * (drag_coefficient/2.0 * self.air_density * self.ref_area * self.velocity.length()**2)

            self.apply_point_force(self.drag_force, self.cp_location)
            self.apply_torque(self.rotational_velocity * -self.friction_coeff * self.air_density)

        # integration

        self.acceleration_local = self.rotation.conj().rotate(self.acceleration)

        self.acceleration.x -= 9.81

        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt

        self.rotational_velocity += self.rotational_acceleration * dt

        ang: float = self.rotational_velocity.length()

        if ang > 0.0:
            self.rotation *= quaternion().from_axis_angle(self.rotational_velocity / ang, ang * dt)

        self.rotation_euler = self.rotation.to_euler()    

        if self.position.x < 0.0:
            self.position.x = 0.0
            self.velocity.x = 0.0

    def clear(self):
        self.acceleration = vec3(0.0, 0.0, 0.0)
        self.rotational_acceleration = vec3(0.0, 0.0, 0.0)