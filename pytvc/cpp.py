"""Python facade for the host-only C++ pyTVC backend.

The embedded C++ core lives under ``cpp/include`` and does not depend on
Python. This module imports the optional extension built from that core and
re-exports classes with names that intentionally mirror the existing pure
Python pyTVC API where practical.

The pure-Python modules remain the default import path during the transition:

    from pytvc.rigidBody import Vector3

Use this module when you explicitly want the C++ backend:

    from pytvc.cpp import Vector3, Quaternion, Rocket
"""

from __future__ import annotations

try:
    from ._pytvc_cpp import (  # type: ignore[attr-defined]
        SCALAR_DOUBLE,
        Actor,
        AeroCoefficientModel,
        Airbrake,
        AirbrakeCan,
        AreaModel,
        ConstantAero,
        ConstantArea,
        ConstantPressure,
        Fin,
        FinCan,
        LinkageModel,
        MotorCurve,
        MotorMount,
        PressureModel,
        PythonTelemetrySink,
        Quaternion,
        RigidBody,
        Rocket,
        RocketBody,
        RollDampener,
        ServoModel,
        SimSample,
        SpinCan,
        Status,
        TVCMount,
        TelemetrySink,
        Vec3,
        Vector3,
    )
except ImportError as exc:  # pragma: no cover - depends on optional build artifact
    raise ImportError(
        "pytvc.cpp requires the optional C++ extension. Build/install with "
        "`python -m pip install -e .[bindings]` from the pyTVC repo root."
    ) from exc


__all__ = [
    "SCALAR_DOUBLE",
    "Actor",
    "AeroCoefficientModel",
    "Airbrake",
    "AirbrakeCan",
    "AreaModel",
    "ConstantAero",
    "ConstantArea",
    "ConstantPressure",
    "Fin",
    "FinCan",
    "LinkageModel",
    "MotorCurve",
    "MotorMount",
    "PressureModel",
    "PythonTelemetrySink",
    "Quaternion",
    "RigidBody",
    "Rocket",
    "RocketBody",
    "RollDampener",
    "ServoModel",
    "SimSample",
    "SpinCan",
    "Status",
    "TVCMount",
    "TelemetrySink",
    "Vec3",
    "Vector3",
]
