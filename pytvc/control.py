from .rigidBody import Vector3, Quaternion, RigidBody
from .motor import Motor
import numpy as np

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
    
class TVCMount(Actor):
    
    def __init__(self, motors: Motor|list[Motor], servoTransferFunction: callable = None, linkageFunction: callable = None) -> None:
        """Initializes a new instance of the TVCMount class.

        Args:
            motors (Motor|list[Motor]): The motor or list of motors to be used in the mount.
            servoTransferFunction (callable): The transfer function for the servo.

        """
        
        super().__init__()
        if isinstance(motors, Motor):
            self._motors = [motors]
        else:
            if not all(isinstance(motor, Motor) for motor in motors):
                raise TypeError("All elements in motors must be of type Motor.")
            self._motors = motors
            
        self._ignitionTimes = [-1] * len(self._motors)
        
        if servoTransferFunction is not None and not callable(servoTransferFunction):
            raise TypeError("servoTransferFunction must be a callable.")
        
        if linkageFunction is not None and not callable(linkageFunction):
            raise TypeError("linkageFunction must be a callable.")
        
        self._linkageFunction = linkageFunction
        self._servoTransferFunction = servoTransferFunction
    
        self._angles = Vector3(0.0, 0.0, 0.0)
        self._targetAngles = Vector3(0.0, 0.0, 0.0)

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

    def linkageFunction(self, fn: callable) -> callable:
        """Set the linkage function for the TVCMount.

        Args:
            fn (callable): The linkage function to set.

        Returns:
            callable: The linkage function.
        """
        
        self._linkageFunction = fn
        return self._linkageFunction
    
    def transferFunction(self, fn: callable) -> callable:
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

class RocketBody(Actor):

    def __init__(self, cpLocation: Vector3, CLFunction: callable, CDFunction: callable, SAFunction: callable, presFunction: callable) -> None:
        """Initializes a new instance of the RocketBody class.

        Args:
            cpLocation (Vector3): The center of pressure location.
            CLFunction (callable): The function to calculate lift coefficient.
            CDFunction (callable): The function to calculate drag coefficient.
            SAFunction (callable): The function to calculate the surface area.
            presFunction (callable): The function to calculate the pressure.
        """

        super().__init__()
        self._cpLocation = cpLocation

        if not callable(CLFunction):
            raise TypeError("CLFunction must be a callable.")
        self._CLFunction = CLFunction

        if not callable(CDFunction):
            raise TypeError("CDFunction must be a callable.")
        self._CDFunction = CDFunction

        if not callable(SAFunction):
            raise TypeError("SAFunction must be a callable.")
        self._SAFunction = SAFunction

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

            aoa = airVelocity.angleBetween(body.rotation.rotate(Vector3(1.0, 0.0, 0.0)))
            if aoa > np.pi / 2:
                aoa = np.pi - aoa

            cd = self._CDFunction(aoa, velMagnitude)
            cl = self._CLFunction(aoa, velMagnitude)
            sa = self._SAFunction(aoa, velMagnitude)
            pres = self._presFunction(aoa, velMagnitude)

            # Calculate lift and drag forces
            lift = 0.5 * cl * sa * pres * velMagnitude**2
            drag = 0.5 * cd * sa * pres * velMagnitude**2

            # Calculate lift and drag vectors in world coordinates

            self._liftVec = airVelocity.cross( body.rotation.rotate(Vector3(1.0, 0.0, 0.0))).cross(airVelocity).norm() * lift
            self._liftVec = body.rotation.rotate(self._liftVec)

            self._dragVec = airVelocity.norm() * -drag
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
