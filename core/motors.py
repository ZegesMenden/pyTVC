import random
from re import T
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

def _load_motor_file(file_name) -> list:
    
    tree = ET.parse(file_name)
    root = tree.getroot()
    
    eng_data = root[0][0][1]

    output = []
    for data in eng_data:
        output.append([float(data.attrib['t']), float(data.attrib['f']), float(data.attrib['m'])])
        
    return output

def _interpolate_thrust(thrust_curve, timeStep):
    thrustList = []
    lPoint = [0, 0, 0]
    for point in thrust_curve:
        if point[0] > 0:
            thrustDiff = point[1] - lPoint[1]
            massDiff = point[2] - lPoint[2]
            timeDiff = point[0] - lPoint[0]
            stepsNeeded = timeDiff * timeStep

            if stepsNeeded > 0:
                adder = thrustDiff / stepsNeeded
                adder_mass = massDiff / stepsNeeded

                i = 0

                thrustToAdd = lPoint[1]
                massToAdd = lPoint[2]

                while i < stepsNeeded:
                    i += 1
                    thrustToAdd += adder
                    if thrustToAdd < 0.0:
                        thrustToAdd = 0.0
                    massToAdd += adder_mass
                    if massToAdd < 0.0:
                        massToAdd = 0.0
                    thrustList.append([thrustToAdd, massToAdd])

        lPoint = point
    return thrustList

class rocket_motor:
    
    def __init__(self, time_step) -> None:
        self.time_step = time_step
        self.motors = {}
        self.time = 0.0
        self.current_thrust = 0.0
        self.current_mass = 0.0
        
    def add_motor(self, motor_name: str, motor_path):
        motor_data_raw = _load_motor_file(motor_path)
        motor_data_interpolated = _interpolate_thrust(motor_data_raw, self.time_step)
        self.motors[motor_name] = {"data": motor_data_interpolated, "ignition_time": 0.0, "is_lit": False}

    def light_motor(self, motor_name, time):
        if motor_name in self.motors:
            if self.motors[motor_name]["is_lit"] == False:
                self.motors[motor_name]["ignition_time"] = int(time * self.time_step + 0.78)
                self.motors[motor_name]["is_lit"] = True
    
    def update(self, time):
        
        time_int = int(time * self.time_step)
        for motor in self.motors:
            if self.motors[motor]["is_lit"] == True:
                if time_int - self.motors[motor]["ignition_time"] >= len(self.motors[motor]["data"]) or time_int - self.motors[motor]["ignition_time"] < 0:
                    self.current_thrust = 0.0
                else:
                    self.current_thrust = self.motors[motor]['data'][time_int - self.motors[motor]["ignition_time"]][0]
                    self.current_mass = self.motors[motor]['data'][time_int - self.motors[motor]["ignition_time"]][1]