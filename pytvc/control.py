from .rigidBody import Vector3, Quaternion, RigidBody
from .motor import Motor
import numpy as np

class ControlActor:
    
    def __init__(self):
        pass
    
    def update(self, body: RigidBody, time: float) -> None:
        """Update the control actor with the given rigid body and time.

        Args:
            body (RigidBody): The rigid body to update.
            time (float): The time to update the control actor.
        """
        pass
    
    def getForce(self) -> Vector3:
        """Get the force applied by the control actor.

        Returns:
            Vector3: The force applied by the control actor in the local reference frame to the parent RigidBody.
        """
        return Vector3(0.0, 0.0, 0.0)
    
    def getTorque(self) -> Vector3:
        """Get the torque applied by the control actor.

        Returns:
            Vector3: The torque applied by the control actor in the local reference frame to the parent RigidBody.
        """
        return Vector3(0.0, 0.0, 0.0)
    
class TVCMount(ControlActor):
    
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
        mass = sum([motor.GetMass(self._time - ignitionTime) if ignitionTime != -1 else motor.GetMass(0) for motor, ignitionTime in zip(self._motors, self._ignitionTimes)])
                
        # Apply the servo transfer function ()
        servoOutput = self._servoTransferFunction(body, self._time, self._targetAngles, thrust, mass)
        
        # Apply the linkage function
        self._angles: Quaternion = self._linkageFunction(body, self._time, servoOutput)
        
        thrustVec = self._angles.rotate(Vector3(thrust, 0, 0))

        return thrustVec, mass
