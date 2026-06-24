import numpy as np
from .rigidBody import Vector3, Quaternion, RigidBody
from .actor import Actor
from .telemetry import SimClock, Logger
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

        # The single, authoritative source of simulation time. It is shared by
        # reference with every logger so all telemetry is stamped consistently.
        self._clock: SimClock = SimClock()

        # Telemetry: one logger for the rocket body and one per actor. They are
        # only written to when logging is enabled.
        self._logger: Logger = Logger(self._clock, "rocket")
        self._actorLoggers: dict[Actor, Logger] = {}
        self._actorNames: dict[Actor, str] = {}
        self._logging: bool = False

    def time(self) -> float:
        return self._clock.simTime

    def getClock(self) -> SimClock:
        """Gets the simulation clock driving this rocket.

        Returns:
            SimClock: The authoritative source of simulation time.
        """

        return self._clock

    def enableLogging(self, on: bool = True) -> None:
        """Enables or disables automatic per-step telemetry recording.

        When enabled, ``update`` records the rocket's and every actor's state
        into their loggers each step. When disabled, logging is zero cost.

        Args:
            on (bool): Whether logging should be active.
        """

        self._logging = bool(on)

    def isLogging(self) -> bool:
        """Returns whether automatic telemetry recording is active."""

        return self._logging

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

    def addActor(self, actor: Actor, position: Vector3 = Vector3(0, 0, 0), name: str | None = None):
        """Adds an actor to the rocket.

        A dedicated telemetry logger is created for the actor. Its ``name`` (or a
        generated ``<ClassName><index>`` if omitted) is used as the namespace
        prefix when the actor's log is merged into the rocket's via ``getLog``.

        Args:
            actor (Actor): The actor to add
            position (Vector3): Mounting position of the actor in the body frame
            name (str | None): Optional unique name for telemetry namespacing
        """

        if not isinstance(actor, Actor):
            raise TypeError("actor must be an instance of Actor")

        if actor not in self._actors:
            resolvedName = name if name is not None else f"{type(actor).__name__}{len(self._actors)}"
            if resolvedName in self._actorNames.values():
                raise ValueError(f"actor name <{resolvedName}> must be unique!")

            self._actors.append(actor)
            self._actorPositions.append(position)
            self._actorNames[actor] = resolvedName
            self._actorLoggers[actor] = Logger(self._clock, resolvedName)

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
            self.addActor(actor)

        return actor

    def update(self, dt: float):
        """Updates the rocket and its actors.

        Args:
            dt (float): The time step for the update
        """

        if dt <= 0:
            raise ValueError("dt must be greater than 0")

        self._clock.advance(dt)
        currentTime = self._clock.simTime

        actorMass = 0.0

        for actor in self._actors:
            actor.update(self._rigidBody, currentTime)
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

        if self._logging:
            self._recordState()

    def _recordState(self) -> None:
        """Records the rocket body and every actor's state for this step.

        The rocket's kinematic state goes into ``self._logger`` and each actor
        records into its own logger. Called automatically by ``update`` when
        logging is enabled.
        """

        body = self._rigidBody
        self._logger.logScalar("time", self._clock.simTime)
        self._logger.logVector("position", body.position)
        self._logger.logVector("velocity", body.velocity)
        self._logger.logVector("accel", self._accel)
        self._logger.logQuaternion("rotation", body.rotation)
        self._logger.logEuler("rotation", body.rotation)
        self._logger.logVector("rot_vel", body.rotVel)
        self._logger.logScalar("mass", body.mass)

        for actor in self._actors:
            actor.logState(self._actorLoggers[actor])

    def getLog(self) -> Logger:
        """Builds and returns a single merged logger of all telemetry.

        The rocket body's traces appear at the top level (e.g. ``position_x``)
        and each actor's traces are namespaced under its name (e.g.
        ``TVCMount0/gimbal_pitch``). Call ``toDict()`` or ``writeCSV(path)`` on
        the result to retrieve the data.

        Returns:
            Logger: A new logger containing the merged rocket and actor logs.
        """

        merged = Logger(self._clock, "merged")
        merged.merge(self._logger)
        for actor in self._actors:
            merged.merge(self._actorLoggers[actor], prefix=self._actorNames[actor])
        return merged
