from core.data import data_visualizer
import csv
import os

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

import numpy as np

class plotter:

    """Data visualization class"""

    def __init__(self) -> None:
        """initializes data visualizer"""

        self.data_descriptions: list = []
        self.header: str = ""
        self.viewer: data_visualizer = data_visualizer()

        self.n_plots: int = 0

    def read_header(self, file_name):
        # os.chdir('..')
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

        plot_points = self.viewer.graph_from_csv(graph_points)

        for index, dataPoint in enumerate(plot_points):
            if index > 0:
                if annotate:
                    plt.plot(plot_points[0], dataPoint,
                             label=graph_points[index])
                else:
                    plt.plot(plot_points[0], dataPoint)
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

        plot_points = self.viewer.graph_from_csv(graph_points)

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

        plot_position = self.viewer.graph_from_csv(graph_points)

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
