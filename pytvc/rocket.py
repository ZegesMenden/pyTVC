import numpy as np
from .rigidBody import Vector3, Quaternion, RigidBody


class Rocket:

    def __init__(
        self,
        dryMass: float,
        inertia: Vector3,
        position: Vector3,
        velocity: Vector3,
        rotation: Quaternion,
        rotVel: Vector3,
    ):
        """Initializes a new instance of the Rocket class.

        Args:
            dryMass (float): Dry mass of the rocket
            inertia (Vector3): Inertia tensor of the rocket
            position (Vector3): Initial position of the rocket
            velocity (Vector3): Initial velocity of the rocket
            rotation (Quaternion): Initial rotation of the rocket
            rotVel (Vector3): Initial rotational velocity of the rocket
        """

        self._dryMass = dryMass
        self._inertia = inertia
        self._rigidBody = RigidBody(
            dryMass, inertia, position, velocity, rotation, rotVel
        )

    def getDryMass(self) -> float:
        """Gets the dry mass of the rocket.

        Returns:
            float: The dry mass of the rocket
        """

        return self._dryMass

    def getInertia(self) -> Vector3:
        """Gets the inertia tensor of the rocket.

        Returns:
            Vector3: The inertia tensor of the rocket
        """

        return self._inertia

    def getRigidBody(self) -> RigidBody:
        """Gets the rigid body of the rocket.

        Returns:
            RigidBody: The rigid body of the rocket
        """

        return self._rigidBody
