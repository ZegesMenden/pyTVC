# pyTVC C++ Core

This directory contains the embedded-oriented C++ implementation of the pyTVC
simulation kernel. The core design rule is that simulation code must not perform
dynamic allocation. Object ownership is external, collections are fixed-capacity,
and platform features are exposed through replaceable interfaces.

Current constraints:

- no `std::vector`, heap ownership pointers, `std::function`, `new`, or `delete`
  in `cpp/include`
- scalar precision selected with `PYTVC_SCALAR_DOUBLE`
- core output goes through `OutputSink`
- telemetry goes through `TelemetrySink`
- motor data is supplied as fixed-capacity `MotorCurve<N>` tables
- embedded smoke build is checked with `-ffreestanding -fno-exceptions -fno-rtti`

Example embedded compile check:

```sh
arm-none-eabi-g++ -std=c++17 -Icpp/include -DPYTVC_SCALAR_DOUBLE=0 \
  -mcpu=cortex-m33 -mthumb -ffreestanding -fno-exceptions -fno-rtti \
  -c cpp/tests/embedded_compile.cpp -o cpp/build/embedded_compile_float.o
```

The Python bindings should wrap this API from the outside. They should not add
requirements to the simulation core.

## Python Binding Layer

The optional host binding module is `pytvc._pytvc_cpp`, built from
`cpp/bindings/bindings.cpp` through the repository root `setup.py`. The user
facing facade is:

```python
from pytvc.cpp import Vector3, Quaternion, RigidBody, Rocket
```

The binding layer intentionally uses host-side ownership (`shared_ptr`) and
pybind11. Those choices are isolated to `cpp/bindings` and are not part of the
embedded core.

Build from the repository root:

```sh
python -m pip install -e .[bindings]
```

On Windows, Python extension builds require Microsoft C++ Build Tools matching
the running Python installation. The embedded ARM compile checks do not require
MSVC.
