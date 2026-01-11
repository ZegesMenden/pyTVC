import numpy as np
from .rigidBody import Vector3, Quaternion, RigidBody
from .actor import Actor
from loguru import logger

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

        self._dryMass: float = dryMass
        self._inertia: Vector3 = inertia
        self._rigidBody: RigidBody = RigidBody(
            dryMass, inertia, position, velocity, rotation, rotVel
        )

        self._actors: list[Actor] = []
        self._actorPositions: list[Vector3] = []

        self._accel: Vector3 = Vector3(0, 0, 0)
        
        self._time: float = 0.0

    def getAccel(self) -> Vector3:
        """Gets the acceleration of the rocket.
        Returns:
            Vector3: The acceleration of the rocket
        """

        return self._accel

    def getDryMass(self) -> float:
        """Gets the dry mass of the rocket.

        Returns:
            float: The dry mass of the rocket
        """

        return self._dryMass
    
    def setDryMass(self, dryMass: float):
        """Sets the dry mass of the rocket.

        Args:
            dryMass (float): The new dry mass of the rocket
        """

        if dryMass <= 0:
            raise ValueError("dryMass must be greater than 0")

        self._dryMass = dryMass

    def getInertia(self) -> Vector3:
        """Gets the inertia tensor of the rocket.

        Returns:
            Vector3: The inertia tensor of the rocket
        """

        return self._inertia
    
    def setInertia(self, inertia: Vector3):
        """Sets the inertia tensor of the rocket.

        Args:
            inertia (Vector3): The new inertia tensor of the rocket
        """

        if not isinstance(inertia, Vector3):
            raise TypeError("inertia must be an instance of Vector3")

        self._inertia = inertia

    def getRigidBody(self) -> RigidBody:
        """Gets the rigid body of the rocket.

        Returns:
            RigidBody: The rigid body of the rocket
        """

        return self._rigidBody

    def addActor(self, actor: Actor, position: Vector3 = Vector3(0, 0, 0)):
        """Adds an actor to the rocket.

        Args:
            actor (Actor): The actor to add
        """
        
        if not isinstance(actor, Actor):
            raise TypeError("actor must be an instance of Actor")

        if actor not in self._actors:
            self._actors.append(actor)
            self._actorPositions.append(position)

    def actor(self, actor: Actor):
        """Gets an actor from the rocket.

        Args:
            actor (Actor): The actor to get

        Returns:
            Actor: The actor from the rocket
        """

        if not isinstance(actor, Actor):
            raise TypeError("actor must be an instance of Actor")

        if actor not in self._actors:
            self._actors.append(actor)

        return actor

    def update(self, dt: float):
        """Updates the rocket and its actors.

        Args:
            dt (float): The time step for the update
        """

        if dt <= 0:
            raise ValueError("dt must be greater than 0")
        
        self._time += dt

        actorMass = 0.0

        for actor in self._actors:
            actor.update(self._rigidBody, self._time)
            if np.isnan(actor.getMass()) or np.isinf(actor.getMass()):
                logger.error(f"Actor object {actor} has an invalid mass of {actor.getMass()}")
            else:
                actorMass += actor.getMass()

        self._rigidBody.mass = self._dryMass + actorMass

        for actor, position in zip(self._actors, self._actorPositions):
            if any([np.isnan(x) or np.isinf(x) for x in iter(actor.getForce())]):
                logger.error(f"Actor <{actor}> returned invalid force of {actor.getForce()}")
            else:
                self._rigidBody.applyLocalForce(actor.getForce(), position)

            if any([np.isnan(x) or np.isinf(x) for x in iter(actor.getTorque())]):
                logger.error(f"Actor <{actor}> returned invalid torque of {actor.getForce()}")
            else:
                self._rigidBody.applyLocalTorque(actor.getTorque())

        self._rigidBody.applyForce(Vector3(-9.806 * self._rigidBody.mass, 0, 0), Vector3(0, 0, 0))

        if self._rigidBody.position.x <= 0:
            self._rigidBody.position = Vector3(0, self._rigidBody.position.y, self._rigidBody.position.z)
            self._rigidBody.velocity = Vector3(0, 0, 0)
            self._rigidBody._accel.x = max(self._rigidBody._accel.x, 0)

        self._accel = self._rigidBody._accel

        self._rigidBody.update(dt)
