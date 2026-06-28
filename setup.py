from __future__ import annotations

from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class Pybind11Extension(Extension):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BuildExt(build_ext):
    c_opts = {
        "msvc": ["/std:c++17", "/EHsc"],
        "unix": ["-std=c++17"],
    }

    def build_extensions(self):
        import pybind11

        include_dirs = [
            str(Path(__file__).parent / "cpp" / "include"),
            pybind11.get_include(),
        ]
        for ext in self.extensions:
            ext.include_dirs.extend(include_dirs)
            compiler_type = self.compiler.compiler_type
            ext.extra_compile_args = self.c_opts.get(compiler_type, ["-std=c++17"])
        super().build_extensions()


ext_modules = [
    Pybind11Extension(
        "pytvc._pytvc_cpp",
        sources=[str(Path("cpp") / "bindings" / "bindings.cpp")],
        language="c++",
    )
]

setup(
    name="pytvc",
    version="0.1",
    author="Cameron Kullberg",
    author_email="camkullberg@gmail.com",
    description="Python package for simulating amateur rockets and control systems",
    packages=["pytvc"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    license="MIT",
    install_requires=[
        "numpy>=2.1.3",
        "loguru>=0.7.3"

    ],
    extras_require={
        # Reference libraries used only by the math cross-check tests
        # (pytvc/test_math_reference.py). Install with: pip install -e .[tests]
        "tests": ["scipy>=1.11"],
        "bindings": ["pybind11>=2.12"],
    },
)
