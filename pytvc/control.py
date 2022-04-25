from pytvc.physics import Vec3
import numpy as np



class PID:

    """PID controller class
    
    for more information on the PID controller, see:
    https://en.wikipedia.org/wiki/PID_controller"""

    def __init__(self, Kp: float = 0.0, Ki: float = 0.0, Kd: float = 0.0, setpoint: float = 0.0, i_max: float = 0.0) -> None:
        """__init__ initializes the PID controller

        Args:
            Kp (float, optional): the proportional gain of the controller. Defaults to 0.0.
            Ki (float, optional): the integral gain of the controller. Defaults to 0.0.
            Kd (float, optional): the derivative gain of the controller. Defaults to 0.0.
            setpoint (float, optional): the setpoint of the controller. Defaults to 0.0.
            i_max (float, optional): the maximum value for the integral component of the controller. Defaults to 0.0.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.i_max = i_max
        self.setpoint = setpoint
        self.i: float = 0.0
        self.last_error: float = 0.0

        self.output: float = 0.0
    
    def update(self, input: float, dt: float = 1.0) -> None:
        error = self.setpoint - input

        if self.i > self.i_max:
            self.i = self.i_max
        elif self.i < -self.i_max:

            self.i = -self.i_max
        
        d = (error - self.last_error) / dt
        self.last_error = error
        
        self.output = self.Kp * error + self.Ki * self.i + self.Kd * d

    def reset(self) -> None:
        self.i = 0.0
        self.last_error = 0.0
    
    def setSetpoint(self, setpoint: float) -> None:
        """setSetpoint sets the setpoint of the PID controller

        Args:
            setpoint (float): setpoint of the PID controller
        """
        self.setpoint = setpoint
    
    def setKp(self, Kp: float) -> None:
        """setKp sets the proportional gain of the PID controller

        Args:
            Kp (float): proportional gain of the PID controller
        """
        self.Kp = Kp
    
    def setKi(self, Ki: float) -> None:
        """setKi sets the integral gain of the PID controller

        Args:
            Ki (float): integral gain of the PID controller
        """
        self.Ki = Ki
    
    def setKd(self, Kd: float) -> None:
        """setKd sets the derivative gain of the PID controller

        Args:
            Kd (float): derivative gain of the PID controller
        """
        self.Kd = Kd
    
    def setImax(self, i_max: float) -> None:
        """setImax set the maximum integral value for the PID controller

        Args:
            i_max (float): maximum value for the integral
        """
        self.i_max = i_max
    
    def getOutput(self) -> float:
        """getOutput returns the output of the PID controller

        Returns:
            float: output of the PID controller
        """
        return self.output

class torque_PID(PID):

    def __init__(self, Kp: float = 0.0, Ki: float = 0.0, Kd: float = 0.0, setpoint: float = 0.0, i_max: float = 0.0, inertia: float = 1.0, lever_arm: float = 1.0) -> None:
        """__init__ initializes the PID controller

        Args:
            Kp (float, optional): the proportional gain of the controller. Defaults to 0.0.
            Ki (float, optional): the integral gain of the controller. Defaults to 0.0.
            Kd (float, optional): the derivative gain of the controller. Defaults to 0.0.
            setpoint (float, optional): the setpoint of the controller. Defaults to 0.0.
            i_max (float, optional): the maximum value for the integral component of the controller. Defaults to 0.0.
            inertia (float, optional): the inertia to pass into the torque controller. Defaults to 1.0.
            lever_arm (float, optional): the lever arm to pass into the torque controller. Defaults to 1.0.
        """

        super().__init__(Kp, Ki, Kd, setpoint, i_max)
        self.inertia = inertia
        self.lever_arm = lever_arm

    def update(self, input: float, dt: float = 1.0, force: float = 1.0) -> None:
        super().update(input, dt)
        if force != 0.0:
            calcval = super().getOutput() * self.inertia / force / self.lever_arm
            if abs(calcval) > 1.0:
                super().output = 0.0
            else:
                super().output = np.arcsin(calcval)
        else:
            super().output = 0.0