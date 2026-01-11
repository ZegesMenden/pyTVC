import xml.etree.ElementTree as ET
import numpy as np
import io
from loguru import logger


def RaspParser(file: io.TextIOBase):
    """RaspParser parses the RASP file and returns a list of tuples containing time, thrust, and mass values.

    Args:
        file (str|io.TextIOBase): The path to the RASP file or a file-like object.

    Returns:
        list[tuple[float, float]]: A list of tuples containing time, thrust, and mass values.
    """

    try:
        for lidx, line in enumerate(file.readlines()):
            vals = line.strip(" \n").split(" ")
            if len(vals) == 2:
                time, thrust = vals[0], vals[1]

                try:
                    thrust = float(thrust)
                    time = float(time)
                except ValueError:
                    logger.warning(
                        f"Invalid data in RASP file on line {lidx}: |{line.strip('\n')}|. Skipping this line."
                    )
                    continue

                if thrust < 0:
                    logger.warning(f"Thrust value is negative: {thrust}.")

                if time < 0:
                    logger.warning(f"Time value is negative: {time}.")

                yield time, thrust

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return


def RSEParser(file: io.TextIOBase):
    """RSEParser parses the RSE file and returns a list of tuples containing time, thrust, and mass values.

    Args:
        file (str|io.TextIOBase): The path to the RSE file or a file-like object.

    Returns:
        list[tuple[float, float]]: A list of tuples containing time, thrust, and mass values.
    """

    try:
        tree = ET.parse(file)
        root = tree.getroot()
    except ET.ParseError as e:
        logger.error(f"Error parsing XML file: {e}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return

    for data in root[0][0][1]:
        time, thrust, mass = (
            float(data.attrib["t"]),
            float(data.attrib["f"]),
            float(data.attrib["m"]),
        )

        if thrust < 0:
            logger.warning(f"Thrust value is negative: {thrust}. Setting to 0.0.")

        if time < 0:
            logger.warning(f"Time value is negative: {time}. Setting to 0.0.")

        if mass < 0:
            logger.warning(f"Mass value is negative: {mass}. Setting to 0.0.")

        yield time, thrust, mass


class Motor:

    def __init__(self, file: str) -> None:
        """Initializes a new instance of the Motor class.

        Args:
            file (str): The path to the motor file.
        """

        self._points = []

        extension = file.split(".")[-1].lower()
        
        if extension == "rasp":
            try:
                for point in RaspParser(file):
                    self._points.append(
                        {"time": point[0], "thrust": point[1], "mass": 0.0}
                    )
            except Exception as e1:
                logger.error(f"An unexpected error occurred: {e1}")
                return
        elif extension == "rse":
            try:
                for point in RSEParser(file):
                    self._points.append(
                        {"time": point[0], "thrust": point[1], "mass": point[2]}
                    )
            except FileNotFoundError as e:
                logger.error(f"File not found: {e}")
                return
            except Exception as e:

                logger.error(f"An unexpected error occurred: {e}")
        else:
            logger.error(
                f"Unsupported file type: {extension}. Supported types are .rse and .rasp."
            )
            return

    def GetThrust(self, time: float) -> float:
        """GetThrust returns the thrust of the motor at the given time

        Args:
            time (float): The time to get the thrust for.

        Returns:
            float: The thrust of the motor at the given time.
        """

        if time < self._points[0]["time"]:
            return 0.0
        elif time > self._points[-1]["time"]:
            return 0.0
        else:
            for i in range(len(self._points) - 1):
                if self._points[i]["time"] <= time <= self._points[i + 1]["time"]:
                    thrust = np.interp(
                        time,
                        [self._points[i]["time"], self._points[i + 1]["time"]],
                        [self._points[i]["thrust"], self._points[i + 1]["thrust"]],
                    )
                    return thrust
            return 0.0

    def GetMass(self, time: float) -> float:
        """GetMass returns the mass of the motor at the given time

        Args:
            time (float): The time to get the mass for.

        Returns:
            float: The mass of the motor at the given time.
        """

        if time < self._points[0]["time"]:
            return 0.0
        elif time > self._points[-1]["time"]:
            return 0.0
        else:
            for i in range(len(self._points) - 1):
                if self._points[i]["time"] <= time <= self._points[i + 1]["time"]:
                    mass = np.interp(
                        time,
                        [self._points[i]["time"], self._points[i + 1]["time"]],
                        [self._points[i]["mass"], self._points[i + 1]["mass"]],
                    )
                    return mass
            return 0.0

    def GetBurnoutTime(self) -> float:
        """GetBurnoutTime returns the burnout time of the motor

        Returns:
            float: The burnout time of the motor.
        """

        return self._points[-1]["time"]

    def getAllPoints(self) -> list:
        """getAllPoints returns all points of the motor

        Returns:
            list: A list of all points of the motor.
        """

        return self._points
