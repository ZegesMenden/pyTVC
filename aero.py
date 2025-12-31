from pytvc.rigidBody import Vector3, Quaternion, RigidBody
from collections.abc import Callable
import numpy as np

class AeroComponent:

    def __init__(self, position: Vector3, dragFunction: Callable[[float, float, RigidBody], float], liftFunction: Callable[[float, float, RigidBody], float]):

        if not isinstance(position, Vector3):
            raise ValueError("position must be a Vector3 object")

        if not callable(dragFunction):
            raise ValueError("dragFunction must be a callable function")
        
        if not callable(liftFunction):
            raise ValueError("liftFunction must be a callable function")

        self._position = position
        self._dragFunction = dragFunction
        self._liftFunction = liftFunction

    @property
    def position(self) -> Vector3:
        return self._position
    
    @position.setter
    def position(self, value: Vector3):
        if not isinstance(value, Vector3):
            raise ValueError("position must be a Vector3 object")
        self._position = value

    @property
    def dragFunction(self) -> Callable[[float, float, RigidBody], float]:
        return self._dragFunction
    
    @property
    def liftFunction(self) -> Callable[[float, float, RigidBody], float]:
        return self._liftFunction
    
    @dragFunction.setter
    def dragFunction(self, value: Callable[[float, float, RigidBody], float]):
        if not callable(value):
            raise ValueError("dragFunction must be a callable function")
        self._dragFunction = value

    @liftFunction.setter
    def liftFunction(self, value: Callable[[float, float, RigidBody], float]):
        if not callable(value):
            raise ValueError("liftFunction must be a callable function")
        self._liftFunction = value

    def getForces(self, state: RigidBody) -> Vector3:
        return Vector3()

class Fin(AeroComponent):

    def __init__(self, position: Vector3, dragFunction: Callable[[float, float, RigidBody], float], liftFunction: Callable[[float, float, RigidBody], float], area: float, angle: float):

        super().__init__(position, dragFunction, liftFunction)
        self._area: float = area
        self._angle: float = 0.0
        self._bodyAngle: float = angle

    def setAngle(self, angle: float):
        self._angle = angle

    def getForces(self, state: RigidBody) -> Vector3:

        # TODO: add Reynolds number based lookup instead of fixed CL/CD callbacks.

        # Fin velocity in body frame (linear + rotational components).
        velocity_body = state.rotation.conjugate().rotate(
            state.velocity + state.rotVel.cross(self.position)
        )

        # Orientation of the fin: body mounting angle then commanded deflection.
        finRotation: Quaternion = (
            Quaternion.fromEulerAngles(Vector3(self._bodyAngle, 0, 0))
            * Quaternion.fromEulerAngles(Vector3(0, 0, self._angle))
        )

        # Velocity in the fin frame.
        velocity_fin = finRotation.conjugate().rotate(velocity_body)
        flowSpeed = abs(velocity_fin)
        if flowSpeed == 0:
            return Vector3()

        angleOfAttack = np.arctan2(-velocity_fin.z, velocity_fin.x)

        # Aerodynamic coefficients from user-supplied models.
        CL = self._liftFunction(angleOfAttack, flowSpeed, state)
        CD = self._dragFunction(angleOfAttack, flowSpeed, state)

        # Directions in the fin frame.
        flowDir = velocity_fin.norm()
        dragDirection = flowDir * -1.0

        # Span axis is +Y in the fin frame; lift is perpendicular to flow and span.
        span_axis = Vector3(0.0, 1.0, 0.0)
        liftDirection = flowDir.cross(span_axis).cross(flowDir).norm()
        if abs(angleOfAttack) > 0:
            liftDirection = liftDirection * np.sign(angleOfAttack)

        # TODO: change this with the actual air density.
        airDensity = 1.225

        aeroCoeff = 0.5 * airDensity * (flowSpeed ** 2) * self._area

        lift = liftDirection * aeroCoeff * CL
        drag = dragDirection * aeroCoeff * CD

        # Rotate back to world frame.
        total_fin_frame = lift + drag
        total_world = state.rotation.rotate(finRotation.rotate(total_fin_frame))

        return total_world
    


