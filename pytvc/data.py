import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import xml.etree.ElementTree as ET
import numpy as np
import os
import csv

# from pytvc.physics import Vec3, Quat

def progress_bar(n, maxn) -> str:
    progress: str = ""
    end = int(n/maxn*50)
    for i in range(50): 
        if i > end: 
            progress += ' ' 
        else: 
            progress += "="
    return (f"""[{progress}] Progress: {round((n/maxn)*100, 2)}%""")

class data_logger:
    def __init__(self):
        self.variables = 0
        self.variableDescriptions = []
        self.currentLog = {}
        self.loggedData = []
        self.initialized = False
        self.fileName = ""
        pass

    def add_datapoint(self, variableName):
        """Adds a data point to the logger object. Datapoints are added sequentially, so add your variables in the same sequence that you want them to show up in on the CSV"""
        if self.initialized == False:
            if str(variableName) in self.currentLog:
                raise IndexError("datapoiont already initialized")
            else:
                self.variables += 1
                self.variableDescriptions.append(variableName)
                self.currentLog[variableName] = None
        else:
            raise IndexError("file already initialized!")

    def record_variable(self, variableName, data):
        """records a variable to the current log, DOES NOT LOG AUTOMATICALLY"""
        if str(variableName) in self.currentLog:
            # if self.currentLog[str(variableName)] != None:
            #     raise Warning(f'data point {str(variableName)} is being overwritten!')
            self.currentLog[str(variableName)] = data
        else:
            raise IndexError("datapoint not initialized")

    def initialize_csv(self, makeFile, overWrite):
        """Initializes the CSV file and prepares it for writing."""
        self.initialized = True

        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        if os.path.exists(str(self.fileName)):

            f = open(str(self.fileName), "r")

            # if the file is empty?
            if not f.read():
                f.close()

                f = open(str(self.fileName), "w")
                outString = ""
                for varName in self.variableDescriptions:
                    outString += varName
                    outString += ","

                f.write(outString[0:-1])

                f.write('\n')
                f.close()
            else:
                if overWrite == True:
                    f.close()

                    f = open(str(self.fileName), "w")
                    outString = ""
                    for varName in self.variableDescriptions:
                        outString += varName
                        outString += ","

                    f.write(outString[0:-1])

                    f.write('\n')
                if overWrite == False:
                    raise OSError("csv file is not empty!")

        else:
            if makeFile == True:
                f = open(str(self.fileName), "w")
                outString = ""
                for varName in self.variableDescriptions:
                    outString += varName
                    outString += ","

                f.write(outString[0:-1])

                f.write('\n')
                f.close()
            else:
                raise OSError("csv file not found!")

    def save_data(self, clearData):
        outString = ""
        for datapoint in self.currentLog:
            currentVar = self.currentLog[str(datapoint)]
            if currentVar == None:
                outString += "0"
            else:
                outString += str(currentVar)
            outString += ","
            if clearData == True:
                self.currentLog[str(datapoint)] = None
        f = open(str(self.fileName), "a")
        f.write(outString[0:-1] + "\n")

    def get_var(self, variableName):
        if str(variableName) in self.currentLog:
            return self.currentLog[str(variableName)]
        else:
            raise IndexError("datapoint not initialized")

class data_visualizer:
    def __init__(self):
        self.allDataDescriptions = []

    def graph_from_csv(self, datapoints, file_name):
        descriptionNum = 0
        pointsToLog = []

        for description in self.allDataDescriptions:
            for requestedDatapoint in datapoints:
                if str(description) == str(requestedDatapoint):
                    pointsToLog.append(descriptionNum)
            descriptionNum += 1

        with open(file_name, newline='\n') as pathFile:
            reader = csv.reader(pathFile, delimiter=',', quotechar='"')
            logList = []
            dataOut = []
            for index, row in enumerate(reader):
                for point in pointsToLog:
                    logList.append(row[point])
                if index == 0:
                    for x in logList:
                        dataOut.append([])

                if index > 0:
                    for idx, point in enumerate(dataOut):
                        point.append(float(row[pointsToLog[idx]]))
                logList = []

        return dataOut

class plotter:

    """Data visualization class"""

    def __init__(self) -> None:
        """initializes data visualizer"""

        self.data_descriptions: list = []
        self.header: str = ""
        self.viewer: data_visualizer = data_visualizer()

        self.n_plots: int = 0

    def read_header(self, file_name):
        self.file_name = file_name
        f = open(file_name, "r")
        self.header = f.readline()
        self.data_descriptions = self.header.split(',')
        self.viewer.allDataDescriptions = self.data_descriptions

    def create_2d_graph(self, graph_points: list, x_desc: str = "", y_desc: str = "", annotate: bool = False):
        """Creates a 2-dimensional graph from the provided data descriptions.

        Params:

        graph_points - a list of all data point names to be graphed. First element of graph_points will be the x-axis datapoint.

        x_desc - label for the x axis of the graph.

        y_desc - label for the y axis of the graph.

        annotate - set true to show a color code for all lines on the graph."""

        self.n_plots += 1
        plt.figure(self.n_plots)

        plot_points = self.viewer.graph_from_csv(graph_points, self.file_name)

        for index, dataPoint in enumerate(plot_points):
            if index > 0:
                if annotate:
                    plt.plot(plot_points[0], dataPoint,
                             label=graph_points[index])
                else:
                    plt.plot(plot_points[0], dataPoint)
        if annotate:
           plt.legend()
        plt.xlabel(x_desc)
        plt.ylabel(y_desc)

    def create_3d_graph(self, graph_points: list, size: float = 0.0, color: str = 'Blues'):
        """Creates a 3 dimensional graph from the provided data descriptions

        Params:

        graph_points - a list of all data point names to be graphed.

        size - the size of the graph in all axes"""

        if len(graph_points) != 3:
            raise IndexError("graph_points must have 3 data points!")

        self.n_plots += 1
        plt.figure(self.n_plots)

        plot_points = self.viewer.graph_from_csv(graph_points, self.file_name)

        ax = plt.axes(projection='3d')

        if size != 0.0:

            ax.set_xlim3d(-size, size)
            ax.set_ylim3d(-size, size)
            ax.set_zlim3d(0, size)

        ax.scatter3D(plot_points[2], plot_points[1],
                     plot_points[0], c=plot_points[2], cmap=color)

    def create_3d_animation(self, graph_points: list, size: float, time: float = 5.0, color: str = 'Blues'):
        
        self.n_plots += 1
        fig = plt.figure(self.n_plots)
        
        ax = p3.Axes3D(fig)

        plot_position = self.viewer.graph_from_csv(graph_points, self.file_name)

        ax.set_xlim3d(-size, size)
        ax.set_ylim3d(-size, size)
        ax.set_zlim3d(0, size)

        def func(num, dataSet, line):
            # NOTE: there is no .set_data() for 3 dim data...
            xcur = dataSet[2][num] - 1
            numpog = 0
            for index, x in enumerate(dataSet[2]):
                if x >= xcur-0.1 and x <= xcur -0.1:
                    numpog = index
            line.set_data(dataSet[0:2, num-20:num])
            line.set_3d_properties(dataSet[2, num-20:num])
            return line

        # # THE DATA POINTS
        t = np.array(plot_position[1]) # This would be the z-axis ('t' means time here)
        x = np.array(plot_position[2])
        y = np.array(plot_position[3])

        numDataPoints = len(plot_position[0])
        dataSet = np.array([x, y, t])

        # NOTE: Can't pass empty arrays into 3d version of plot()
        line = plt.plot(dataSet[0], dataSet[1], dataSet[2], lw=2, c='g')[0] # For line plot

        ax.set_xlabel('y')
        ax.set_ylabel('z')
        ax.set_zlabel('x')
        ax.set_title('Trajectory')

        # Creating the Animation object
        line_ani = animation.FuncAnimation(fig, func, frames=numDataPoints, fargs=(dataSet,line), interval=(time/numDataPoints), blit=False)
        plt.show()
    def show_all_graphs(self):
        """Displays all graphs."""
        plt.show()

class rocket_motor:

    """class representing a rocket motor"""

    def __init__(self, filePath: str, timeStep: int, ignitionTime: float = -1.0) -> None:
        """__init__ initializes the motor object

        Args:
            filePath (str, optional): the path to the motor file. Defaults to "".
            timeStep (int, optional): number of data points per second. Defaults to 0.
            ignitionTime (float, optional): ignition time of the motor. Defaults to -1.0.
        """
        self._ignitionTime: float = ignitionTime
        self._timeStep: int = timeStep
        self._data = []
        
        if filePath != "":
            
            tree = ET.parse(filePath)
            root = tree.getroot()
            
            eng_data = root[0][0][1]
            
            lPoint = [0, 0, 0]
            
            for data in eng_data:
                dataTmp = [float(data.attrib['t']), float(data.attrib['f']), float(data.attrib['m'])]
                if dataTmp[0] > 0:
                    thrustDiff = dataTmp[1] - lPoint[1]
                    massDiff = dataTmp[2] - lPoint[2]
                    timeDiff = dataTmp[0] - lPoint[0]
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
                            self._data.append([thrustToAdd, massToAdd])
                lPoint = dataTmp

    def set_ignition_time(self, time: float) -> None:
        """set_ignition_time: sets the ignition time of the motor

        Args:
            time (float): the ignition time
        """
        self._ignitionTime = time

    def update(self, time: float) -> tuple[float, float]:
        """update: calculates and returns the motor's thrust and mass values based on the time passed in

        Args:
            time (float): current time

        Returns:
            tuple (float, float): thrust and mass values
        """
        if time > self._ignitionTime and self._ignitionTime != -1.0:
            idx = int(time*self._timeStep)
            if idx < len(self._data):
                return self._data[idx][0], self._data[idx][1]*0.001
            else:
                return 0.0, self._data[-1][1]*0.001
        else:
            return 0.0, self._data[0][1]*0.001