
from __future__ import annotations

import csv
import time
from enum import Enum

from .rigidBody import Vector3, Quaternion


class SimClock:
    """The single, authoritative source of simulation time.

    A `SimClock` is owned by the `Rocket` and shared (by reference) with every
    `Logger` so that all telemetry is stamped against one consistent timeline.
    `simTime` advances only when the owner calls `advance(dt)`; `realTime` is
    wall-clock seconds since the clock started (or was last reset).
    """

    def __init__(self) -> None:
        self._simTime: float = 0.0
        self._realStart: float = time.perf_counter()

    def advance(self, dt: float) -> None:
        """Advance simulation time by `dt` seconds."""
        if dt <= 0:
            raise ValueError("dt must be greater than 0")
        self._simTime += dt

    def reset(self) -> None:
        """Reset simulation time to zero and restart the real-time reference."""
        self._simTime = 0.0
        self._realStart = time.perf_counter()

    @property
    def simTime(self) -> float:
        """Current simulation time in seconds."""
        return self._simTime

    @property
    def realTime(self) -> float:
        """Wall-clock seconds elapsed since this clock started / last reset."""
        return time.perf_counter() - self._realStart


class LogFormat(Enum):

    NUMBER  = 0
    STRING  = 1
    BOOL    = 2

    @staticmethod
    def infer(value: object) -> "LogFormat":
        """Infer the LogFormat for a value (bool is checked before int)."""
        if isinstance(value, bool):
            return LogFormat.BOOL
        if isinstance(value, (int, float)):
            return LogFormat.NUMBER
        if isinstance(value, str):
            return LogFormat.STRING
        raise TypeError(f"cannot infer LogFormat for value of type {type(value)}")


class LogEntry:

    def __init__(self, value: object, sim_time: float, real_time: float) -> None:

        self.__value: object = value
        self.__simTime: float = sim_time
        self.__realTime: float = real_time

    @property
    def value(self) -> object:
        """Get the logged value."""
        return self.__value

    @property
    def simTime(self) -> float:
        """Get the simulation time at which this entry was recorded."""
        return self.__simTime

    # Backwards-compatible alias.
    @property
    def rocketTime(self) -> float:
        """Deprecated alias for `simTime`."""
        return self.__simTime

    @property
    def realTime(self) -> float:
        """Get the real elapsed time at which this entry was recorded."""
        return self.__realTime


class LogTrace:

    def __init__(self, name: str, format: LogFormat = LogFormat.NUMBER) -> None:

        if not isinstance(name, str): raise ValueError("name must be a string!")
        if name == "": raise ValueError("name must not be empty!")
        if name.strip() == "": raise ValueError("name must not be blank!")

        if not isinstance(format, LogFormat): raise ValueError("format must be of type LogFormat!")
        if format not in LogFormat: raise ValueError("format must be a valid LogFormat value!")

        self.__name: str = name
        self.__format: LogFormat = format
        self.__entries: list[LogEntry] = []

    @property
    def name(self) -> str:
        """Get the immutable name of this trace."""
        return self.__name

    @property
    def format(self) -> LogFormat:
        """Get the immutable format of this trace."""
        return self.__format

    def addEntry(self, entry: LogEntry) -> None:
        """Add a LogEntry to this trace."""
        if not isinstance(entry, LogEntry):
            raise TypeError("entry must be of type LogEntry!")
        self.__entries.append(entry)

    def get_entries(self) -> list[LogEntry]:
        """Get a copy of all logged entries. Returns a read-only copy."""
        return self.__entries.copy()

    def renamed(self, name: str) -> "LogTrace":
        """Return a copy of this trace under a new name.

        The copy shares the same `LogEntry` objects (entries are immutable), so
        this is cheap; it is used by `Logger.merge` to namespace traces.
        """
        clone = LogTrace(name, self.__format)
        for entry in self.__entries:
            clone.addEntry(entry)
        return clone


class Logger:

    def __init__(self, clock: SimClock, name: str = "") -> None:

        if not isinstance(clock, SimClock):
            raise TypeError("clock must be of type SimClock!")
        if not isinstance(name, str):
            raise TypeError("name must be a string!")

        self.__clock: SimClock = clock
        self.__name: str = name
        self.__traces: dict[str, LogTrace] = {}

    @property
    def name(self) -> str:
        """Get the name of this logger."""
        return self.__name

    @property
    def clock(self) -> SimClock:
        """Get the SimClock this logger is stamped against."""
        return self.__clock

    def traceNames(self) -> list[str]:
        """Get the names of all traces in this logger."""
        return list(self.__traces.keys())

    def addTrace(self, name: str, traceFormat: LogFormat = LogFormat.NUMBER) -> LogTrace:

        if not isinstance(name, str): raise ValueError("name must be a string!")
        if name == "": raise ValueError("name must not be empty!")
        if name.strip() == "": raise ValueError("name must not be blank!")

        if name in self.__traces.keys(): raise ValueError(f"name <{name}> must be unique!")

        trace = LogTrace(name, traceFormat)
        self.__traces[name] = trace
        return trace

    def log(self, name: str, value: object) -> None:

        if not isinstance(name, str): raise ValueError("name must be a string!")
        if name not in self.__traces.keys(): raise ValueError(f"name <{name}> must exist in the logger to be logged!")

        trace: LogTrace = self.__traces[name]

        if trace.format == LogFormat.NUMBER and not isinstance(value, (int, float)):
            raise TypeError(f"value for <{name}> must be a number!")
        if trace.format == LogFormat.STRING and not isinstance(value, str):
            raise TypeError(f"value for <{name}> must be a string!")
        if trace.format == LogFormat.BOOL and not isinstance(value, bool):
            raise TypeError(f"value for <{name}> must be a bool!")

        entry: LogEntry = LogEntry(value, self.__clock.simTime, self.__clock.realTime)
        trace.addEntry(entry)

    def logAuto(self, name: str, value: object) -> None:
        """Log a value, creating the trace on first use with an inferred format.

        This is the convenience entry point used by actor `logState` methods so
        they do not have to declare every trace up front. `bool` values are
        treated as `LogFormat.BOOL` (not `NUMBER`).
        """
        fmt = LogFormat.infer(value)
        if name not in self.__traces:
            self.addTrace(name, fmt)
        # Normalise numpy scalars / ints so strict `log` validation passes.
        if fmt == LogFormat.NUMBER and not isinstance(value, bool):
            self.log(name, float(value))  # type: ignore[arg-type]
        else:
            self.log(name, value)

    def logScalar(self, name: str, value: float) -> None:
        """Log a single scalar value."""
        self.logAuto(name, float(value))

    def logVector(self, name: str, vec: Vector3) -> None:
        """Log a Vector3 as `name_x`, `name_y`, `name_z`."""
        self.logScalar(f"{name}_x", vec.x)
        self.logScalar(f"{name}_y", vec.y)
        self.logScalar(f"{name}_z", vec.z)

    def logQuaternion(self, name: str, quat: Quaternion) -> None:
        """Log a Quaternion as `name_w`, `name_x`, `name_y`, `name_z`."""
        self.logScalar(f"{name}_w", quat.w)
        self.logScalar(f"{name}_x", quat.x)
        self.logScalar(f"{name}_y", quat.y)
        self.logScalar(f"{name}_z", quat.z)

    def logEuler(self, name: str, quat: Quaternion) -> None:
        """Log a Quaternion's euler angles as `name_roll/_pitch/_yaw` (radians)."""
        euler = quat.toEulerAngles()
        self.logScalar(f"{name}_roll", euler.x)
        self.logScalar(f"{name}_pitch", euler.y)
        self.logScalar(f"{name}_yaw", euler.z)

    def merge(self, other: "Logger", prefix: str = "") -> None:
        """Fold another logger's traces into this one.

        Every trace from `other` is copied in, renamed to `f"{prefix}/{name}"`
        when a prefix is supplied (e.g. an actor name). Raises on a name
        collision so that data is never silently overwritten.
        """
        if not isinstance(other, Logger):
            raise TypeError("other must be of type Logger!")

        for name in other.traceNames():
            newName = f"{prefix}/{name}" if prefix else name
            if newName in self.__traces:
                raise ValueError(f"merge collision: trace <{newName}> already exists!")
            self.__traces[newName] = other._getTrace(name).renamed(newName)

    def _getTrace(self, name: str) -> LogTrace:
        """Internal: fetch a trace by name (used by `merge`)."""
        return self.__traces[name]

    def clear(self) -> None:
        """Drop all traces and their entries, so the logger can be reused."""
        self.__traces.clear()

    def toDict(self) -> dict[str, dict]:
        """Export every trace as plain Python lists.

        Returns a mapping of trace name to a dict with keys `format`,
        `sim_time`, `real_time`, and `values`.
        """
        out: dict[str, dict] = {}
        for name, trace in self.__traces.items():
            entries = trace.get_entries()
            out[name] = {
                "format": trace.format.name,
                "sim_time": [e.simTime for e in entries],
                "real_time": [e.realTime for e in entries],
                "values": [e.value for e in entries],
            }
        return out

    def writeCSV(self, path: str) -> None:
        """Write all traces to a wide CSV keyed by simulation time.

        Columns are `sim_time`, `real_time`, then one column per trace. Rows are
        the sorted union of every trace's sim-time stamps; cells with no sample
        at a given time are left blank. (Actors recorded on the same steps line
        up exactly.)
        """
        names = sorted(self.__traces.keys())

        # value/real_time indexed by (trace name, sim_time).
        byTime: dict[float, dict[str, object]] = {}
        realByTime: dict[float, float] = {}
        for name in names:
            for entry in self.__traces[name].get_entries():
                byTime.setdefault(entry.simTime, {})[name] = entry.value
                realByTime.setdefault(entry.simTime, entry.realTime)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sim_time", "real_time", *names])
            for t in sorted(byTime.keys()):
                row = [t, realByTime.get(t, "")]
                row.extend(byTime[t].get(name, "") for name in names)
                writer.writerow(row)
