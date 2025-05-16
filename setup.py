from setuptools import setup, find_packages

setup(
    name="pytvc",
    version="0.1",
    author="Cameron Kullberg",
    author_email="camkullberg@gmail.com",
    description="Python package for generating 3d solder stencils from gerber files",
    packages=["pytvc"],
    license="MIT",
    install_requires=[
        "numpy>=2.1.3",
        "loguru>=0.7.3"
        
    ],
)
