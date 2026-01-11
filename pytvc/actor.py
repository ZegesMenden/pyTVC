from .rigidBody import Vector3, Quaternion, RigidBody
from collections.abc import Callable
from .motor import Motor
import numpy as np
from loguru import logger

class Actor:
    
    def __init__(self):
        pass
    
    def update(self, body: RigidBody, time: float) -> None:
        """Update the actor with the given rigid body and time.

        Args:
            body (RigidBody): The rigid body to update.
            time (float): The time to update the actor.
        """
        pass
    
    def getForce(self) -> Vector3:
        """Get the force applied by the actor.

        Returns:
            Vector3: The force applied by the actor in the local reference frame to the parent RigidBody.
        """
        return Vector3(0.0, 0.0, 0.0)
    
    def getTorque(self) -> Vector3:
        """Get the torque applied by the actor.

        Returns:
            Vector3: The torque applied by the actor in the local reference frame to the parent RigidBody.
        """
        return Vector3(0.0, 0.0, 0.0)
    
    def getMass(self) -> float:
        """Get the mass of the actor.

        Returns:
            float: The mass of the actor.
        """
        return 0.0

class RollDampener(Actor):

    def __init__(self, dampingCoefficient: Vector3):
        """Initialize a RollDampener class. This Actor applies a damping torque proportional to the current roll rate in each axis.

        Args:
            dampingCoefficient (Vector3): damping coefficient for each axis

        Raises:
            TypeError: if dampingCoefficient is not of type Vector3
        """

        if not isinstance(dampingCoefficient, Vector3):
            raise TypeError("dampingCoefficient must be of type Vector3")
        
        self.__dampingCoefficient: Vector3 = dampingCoefficient
        self.__dampingTorque: Vector3 = Vector3()

    def update(self, body: RigidBody, time: float) -> None:
        """Update the actor with the given rigid body and time.

        Args:
            body (RigidBody): The rigid body to update.
            time (float): The time to update the actor.
        """
        
        self.__dampingTorque = body.rotation.conjugate().rotate(self.__dampingCoefficient * body.rotVel)
    
    def getTorque(self) -> Vector3:
        """Get the torque applied by the actor.

        Returns:
            Vector3: The torque applied by the actor in the local reference frame to the parent RigidBody.
        """
        return self.__dampingTorque

class MotorMount(Actor):
    
    def __init__(self, motors: Motor|list[Motor]) -> None:
        """Initializes a new instance of the TVCMount class.

        Args:
            motors (Motor|list[Motor]): The motor or list of motors to be used in the mount.
        """
        
        super().__init__()
        if isinstance(motors, Motor):
            self._motors = [motors]
        else:
            if not all(isinstance(motor, Motor) for motor in motors):
                raise TypeError("All elements in motors must be of type Motor.")
            self._motors = motors
            
        self._ignitionTimes = [-1.0] * len(self._motors)
        self._mass = sum(motor.GetMass(0) for motor in self._motors)
        
    def igniteMotor(self, motor: Motor|int, time: float) -> None:
        """Ignite the specified motor at the given time.

        Args:
            motor (Motor): The motor to ignite.
            time (float): The time to ignite the motor.
        """
        
        if isinstance(motor, int):

            if motor < 0 or motor >= len(self._motors):
                raise IndexError("Motor index out of range.")
            motor = self._motors[motor]

        if motor not in self._motors:
            raise ValueError("Motor not found in the TVCMount.")
        
        index = self._motors.index(motor)
        self._ignitionTimes[index] = time

    def update(self, body: RigidBody, time: float) -> None:
        """Update the control actor with the given rigid body and time.

        Args:
            body (RigidBody): The rigid body to update.
            time (float): The time to update the control actor.
        """
        
        self._time = time
        
        # Get the thrust from each motor
        thrust = sum([motor.GetThrust(self._time - ignitionTime) if ignitionTime != -1 else motor.GetThrust(0) for motor, ignitionTime in zip(self._motors, self._ignitionTimes)])
        self._mass = sum([motor.GetMass(self._time - ignitionTime) if ignitionTime != -1 else motor.GetMass(0) for motor, ignitionTime in zip(self._motors, self._ignitionTimes)])

        self._thrustVec = Vector3(thrust, 0, 0)

    def getForce(self) -> Vector3:
        """Get the force applied by the control actor.

        Returns:
            Vector3: The force applied by the control actor in the local reference frame to the parent RigidBody.
        """
        
        return self._thrustVec
    
    def getTorque(self) -> Vector3:
        """Get the torque applied by the control actor.

        Returns:
            Vector3: The torque applied by the control actor in the local reference frame to the parent RigidBody.
        """
        
        return Vector3(0.0, 0.0, 0.0)
    
    def getMass(self) -> float:
        """Get the mass of the actor.
        Returns:
            float: The mass of the actor.
        """

        return self._mass / 1000

class TVCMount(MotorMount):
    
    def __init__(self, motors: Motor|list[Motor], servoTransferFunction: Callable[[RigidBody, float, Vector3, float, float], Vector3], linkageFunction: Callable[[RigidBody, float, Vector3], Quaternion]) -> None:
        """Initializes a new instance of the TVCMount class.

        Args:
            motors (Motor|list[Motor]): The motor or list of motors to be used in the mount.
            servoTransferFunction (callable): The transfer function for the servo.

        """
        
        if isinstance(motors, Motor):
            self._motors = [motors]
        else:
            if not all(isinstance(motor, Motor) for motor in motors):
                raise TypeError("All elements in motors must be of type Motor.")
            self._motors = motors
            
        self._ignitionTimes = [-1.0] * len(self._motors)
        
        if servoTransferFunction is not None and not callable(servoTransferFunction):
            raise TypeError("servoTransferFunction must be a callable.")
        
        if linkageFunction is not None and not callable(linkageFunction):
            raise TypeError("linkageFunction must be a callable.")
        
        self._linkageFunction = linkageFunction
        self._servoTransferFunction = servoTransferFunction
    
        self._angles: Quaternion = Quaternion()
        self._targetAngles: Vector3 = Vector3()

        self._mass = sum(motor.GetMass(0) for motor in self._motors)

    def linkageFunction(self, fn: Callable[[RigidBody, float, Vector3], Quaternion]) -> Callable[[RigidBody, float, Vector3], Quaternion]:
        """Set the linkage function for the TVCMount.

        Args:
            fn (callable): The linkage function to set.

        Returns:
            callable: The linkage function.
        """
        
        self._linkageFunction = fn
        return self._linkageFunction
    
    def transferFunction(self, fn: Callable[[RigidBody, float, Vector3, float, float], Vector3]) -> Callable[[RigidBody, float, Vector3, float, float], Vector3]:
        """Set the transfer function for the TVCMount.

        Args:
            fn (callable): The transfer function to set.

        Returns:
            callable: The transfer function.
        """
        
        self._servoTransferFunction = fn
        return self._servoTransferFunction
    
    def setTargetAngles(self, angles: Vector3) -> None:
        """Set the target angles for the TVCMount.

        Args:
            angles (Vector3): The target angles to set.
        """
        
        self._targetAngles = angles
    
    def update(self, body: RigidBody, time: float) -> None:
        """Update the control actor with the given rigid body and time.

        Args:
            body (RigidBody): The rigid body to update.
            time (float): The time to update the control actor.
        """
        
        self._time = time
        
        # Get the thrust from each motor
        thrust = sum([motor.GetThrust(self._time - ignitionTime) if ignitionTime != -1 else motor.GetThrust(0) for motor, ignitionTime in zip(self._motors, self._ignitionTimes)])
        self._mass = sum([motor.GetMass(self._time - ignitionTime) if ignitionTime != -1 else motor.GetMass(0) for motor, ignitionTime in zip(self._motors, self._ignitionTimes)])

        # Apply the servo transfer function ()
        servoOutput = self._servoTransferFunction(body, self._time, self._targetAngles, thrust, self._mass)
        
        # Apply the linkage function
        self._angles: Quaternion = self._linkageFunction(body, self._time, servoOutput)
        
        self._thrustVec = self._angles.rotate(Vector3(thrust, 0, 0))

    def getForce(self) -> Vector3:
        """Get the force applied by the control actor.

        Returns:
            Vector3: The force applied by the control actor in the local reference frame to the parent RigidBody.
        """
        
        return self._thrustVec
    
    def getTorque(self) -> Vector3:
        """Get the torque applied by the control actor.

        Returns:
            Vector3: The torque applied by the control actor in the local reference frame to the parent RigidBody.
        """
        
        return Vector3(0.0, 0.0, 0.0)
    
    def getMass(self) -> float:
        """Get the mass of the actor.
        Returns:
            float: The mass of the actor.
        """

        return self._mass / 1000
    
    def getAngles(self) -> Quaternion:
        """Get the angles of the TVCMount.

        Returns:
            Quaternion: The current angles of the TVCMount.
        """
        
        return self._angles
    
    def getSetpoint(self) -> Vector3:
        """Get the target angles of the TVCMount.

        Returns:
            Vector3: The target angles of the TVCMount as euler angles.
        """
        
        return self._targetAngles

class AeroComponent(Actor):

    def __init__(self, position: Vector3, dragFunction: Callable[[float, float, RigidBody], float], liftFunction: Callable[[float, float, RigidBody], float], presFunction: Callable[[RigidBody], float]):

        if not isinstance(position, Vector3):
            raise ValueError("position must be a Vector3 object")

        if not callable(dragFunction):
            raise ValueError("dragFunction must be a callable function")
        
        if not callable(liftFunction):
            raise ValueError("liftFunction must be a callable function")
    
        if not callable(presFunction):
            raise ValueError("presFunction must be a callable function")

        self._position = position
        self._dragFunction = dragFunction
        self._liftFunction = liftFunction
        self._presFunction = presFunction

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

class Fin(AeroComponent):

    def __init__(self, position: Vector3, dragFunction: Callable[[float, float, RigidBody], float], liftFunction: Callable[[float, float, RigidBody], float], presFunction: Callable[[RigidBody], float], area: float, angle: float):

        super().__init__(position, dragFunction, liftFunction, presFunction)
        self._area: float = area
        self._angle: float = 0.0
        self._bodyAngle: float = angle
        self.liftForces: list[Vector3] = []
        self.dragForces: list[Vector3] = []
        self.__force: Vector3 = Vector3()
        self.__torque: Vector3 = Vector3()

    def setAngle(self, angle: float):
        self._angle = angle

    def getAngle(self) -> float:
        return self._angle

    def getBodyAngle(self) -> float:
        return self._bodyAngle

    def getRotationBody(self) -> Quaternion:
        """Fin rotation relative to the rocket body frame.

        Convention matches `getForces()`: body mounting angle about +X then
        commanded deflection about +Y.
        """
        return (
            Quaternion.fromEulerAngles(Vector3(self._bodyAngle, 0, 0))
            * Quaternion.fromEulerAngles(Vector3(0, self.getAngle(), 0))
        )
    
    def update(self, body: RigidBody, time: float) -> None:
        """Update the actor with the given rigid body and time.

        Args:
            body (RigidBody): The rigid body to update.
            time (float): The time to update the actor.
        """
        
        self.__force = Vector3()
        self.__torque = Vector3()

        # TODO: add Reynolds number based lookup instead of fixed CL/CD callbacks.

        # Fin velocity in body frame (linear + rotational components).
        # Keep frames consistent:
        # - `self.position` is in the rocket/body frame
        # - `body.velocity` and `body.rotVel` are in the world frame
        # Convert world -> body first, then apply omega x r in the body frame.
        bodyVelocity = body.rotation.conjugate().rotate(body.velocity)
        bodyRotVel = body.rotation.conjugate().rotate(body.rotVel)
        bodyVelocity = bodyVelocity + bodyRotVel.cross(self.position)

        # Orientation of the fin: body mounting angle then commanded deflection.
        finRotation: Quaternion = self.getRotationBody()

        # Velocity in the fin frame.
        finVelocity = finRotation.conjugate().rotate(bodyVelocity)
        flowSpeed = abs(finVelocity)
        if flowSpeed == 0:
            return
    
        # print(finVelocity)

        angleOfAttack = np.arctan2(-finVelocity.z, finVelocity.x)

        # print(self.getAngle()*180/np.pi)
        # print(angleOfAttack*180/np.pi)

        # Aerodynamic coefficients from user-supplied models.
        CL = self._liftFunction(angleOfAttack, flowSpeed, body)
        CD = self._dragFunction(angleOfAttack, flowSpeed, body)

        # Directions in the fin frame.
        flowDir = finVelocity.norm()
        dragDirection = flowDir * -1.0

        # Span axis is +Y in the fin frame; lift is perpendicular to flow and span.
        # spanAxis = Vector3(0.0, 0.0, 1.0)
        # liftDirection = flowDir.cross(spanAxis).cross(flowDir).norm()
        spanAxis = Vector3(0.0, 1.0, 0.0)   # if fin-frame y is span
        liftDirection = flowDir.cross(spanAxis).norm()  # perpendicular to flow and span
        if abs(angleOfAttack) > 0:
            liftDirection = liftDirection# * np.sign(angleOfAttack)

        # TODO: change this with the actual air density.
        airDensity = 1.225

        # Find lift and drag
        aeroCoeff = 0.5 * airDensity * (flowSpeed ** 2) * self._area

        lift = liftDirection * aeroCoeff * CL
        drag = dragDirection * aeroCoeff * CD

        bodyLift = finRotation.rotate(lift)
        bodyDrag = finRotation.rotate(drag)
        
        # Rotate forces back to world frame.
        totalForcesFinFrame = lift + drag
        totalForcesBodyFrame = finRotation.rotate(totalForcesFinFrame)

        self.liftForces.append(bodyLift)
        self.dragForces.append(bodyDrag)
        
        self.__force = totalForcesBodyFrame
        self.__torque = Vector3()

    def getForce(self) -> Vector3:
        """Get the force applied by the actor.

        Returns:
            Vector3: The force applied by the actor in the local reference frame to the parent RigidBody.
        """
        return self.__force
    
    def getTorque(self) -> Vector3:
        """Get the torque applied by the actor.

        Returns:
            Vector3: The torque applied by the actor in the local reference frame to the parent RigidBody.
        """
        return Vector3(0.0, 0.0, 0.0)

class FinCan(Actor):

    def __init__(self, position: Vector3, dragFunction: Callable[[float, float, RigidBody], float], liftFunction: Callable[[float, float, RigidBody], float], presFunction: Callable[[RigidBody], float], area: float, radialDistance: float, finCount: int, offsetInitializer: Callable[[], float] = lambda: 0.0):

        self.__fins: list[Fin] = [
            Fin(Quaternion.fromEulerAngles(Vector3(ang, 0, 0)).rotate(Vector3(0, radialDistance, 0)) + position,
                dragFunction,
                liftFunction,
                presFunction,
                area,
                ang) for ang in np.linspace(0, (np.pi*2)*((finCount-1)/finCount), finCount)
        ]

        for fin in self.__fins:
            fin.setAngle(offsetInitializer())

        self.__position = position

    @property
    def position(self) -> Vector3:
        return self.__position

    def getFins(self) -> list[Fin]:
        return self.__fins

    def update(self, body: RigidBody, time: float) -> None:
        
        for fin in self.__fins:
            fin.update(body, time)
    
    def getForce(self) -> Vector3:

        totalForce: Vector3 = Vector3()
        for fin in self.__fins:
            totalForce += fin.getForce()
        
        return totalForce

    def getTorque(self) -> Vector3:

        totalTorque: Vector3 = Vector3()
        for fin in self.__fins:
            totalTorque += fin.getTorque()
            totalTorque += fin.position.cross(fin.getForce()) * Vector3(1, 0, 0)
        
        return totalTorque
    
class SpinCan(FinCan):

    def __init__(self, position: Vector3, dragFunction: Callable[[float, float, RigidBody], float], liftFunction: Callable[[float, float, RigidBody], float], presFunction: Callable[[RigidBody], float], area: float, radialDistance: float, finCount: int, offsetInitializer: Callable[[], float] = lambda: 0.0, rollCoefficient: float = 0.0):

        super().__init__(position, dragFunction, liftFunction, presFunction, area, radialDistance, finCount, offsetInitializer)
        self.__rollCoefficient: float = rollCoefficient

    def getForce(self) -> Vector3:

        totalForce: Vector3 = Vector3()
        for fin in super().getFins():
            totalForce += fin.getForce()
        
        return totalForce

    def getTorque(self) -> Vector3:

        totalTorque: Vector3 = Vector3()
        for fin in super().getFins():
            totalTorque += fin.getTorque()
            forceTorque = fin.position.cross(fin.getForce()) * Vector3(self.__rollCoefficient, 0, 0)
            totalTorque += forceTorque

        return totalTorque

class RocketBody(AeroComponent):

    def __init__(self, position: Vector3, dragFunction: Callable[[float, float, RigidBody], float], liftFunction: Callable[[float, float, RigidBody], float], presFunction: Callable[[RigidBody], float], areaFunction: Callable[[float, float, RigidBody], float]) -> None:

        super().__init__(position, dragFunction, liftFunction, presFunction)

        if not callable(areaFunction):
            raise TypeError("areaFunction must be a callable.")
        self._areaFunction = areaFunction

        if not callable(presFunction):
            raise TypeError("presFunction must be a callable.")
        self._presFunction = presFunction

        self._liftVec = Vector3(0.0, 0.0, 0.0)
        self._dragVec = Vector3(0.0, 0.0, 0.0)

    def update(self, body: RigidBody, time: float) -> None:
        """Update the control actor with the given rigid body and time.

        Args:
            body (RigidBody): The rigid body to update.
            time (float): The time to update the control actor.
        """
        
        self._time = time

        airVelocity = body.velocity
        velMagnitude = airVelocity.len()

        # Calculate angle of attack if velocity is non-zero
        if velMagnitude > 0:

            bodyAxisWorld = body.rotation.rotate(Vector3(1.0, 0.0, 0.0)).norm()
            aoa = airVelocity.angleBetween(bodyAxisWorld)
            if aoa > np.pi / 2:
                aoa = np.pi - aoa

            cd = self._dragFunction(aoa, velMagnitude, body)
            cl = self._liftFunction(aoa, velMagnitude, body)
            sa = self._areaFunction(aoa, velMagnitude, body)
            pres = self._presFunction(body)

            # Aerodynamic force magnitudes.
            lift = 0.5 * cl * sa * pres * velMagnitude**2
            drag = 0.5 * cd * sa * pres * velMagnitude**2

            # Drag always opposes the airflow.
            self._dragVec = airVelocity.norm() * -drag

            # Body lift/normal force should oppose the lateral component of airflow
            # relative to the body axis (restoring, passive stability).
            v_para = bodyAxisWorld * airVelocity.dot(bodyAxisWorld)
            v_perp = airVelocity - v_para
            if v_perp.len() > 1e-12 and abs(lift) > 0.0:
                self._liftVec = v_perp.norm() * (-lift)
            else:
                self._liftVec = Vector3()

        else:
            self._liftVec = Vector3(0.0, 0.0, 0.0)
            self._dragVec = Vector3(0.0, 0.0, 0.0)

    def getForce(self) -> Vector3:
        return self._liftVec + self._dragVec
    
    def getTorque(self) -> Vector3:
        return Vector3(0.0, 0.0, 0.0)
    
    def getLift(self) -> Vector3:
        """Get the lift vector.

        Returns:
            Vector3: The lift vector.
        """
        
        return self._liftVec
    
    def getDrag(self) -> Vector3:
        """Get the drag vector.

        Returns:
            Vector3: The drag vector.
        """

        return self._dragVec
