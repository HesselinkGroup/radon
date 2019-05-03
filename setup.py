from setuptools import find_packages
from setuptools import setup

requirements = ["pytest", "scipy", "numpy"]
version = "0.0.1"

setup(
    name="radon",
    version=version,
    author="Paul Hansen",
    description="CT simulation tools",
    packages=find_packages(exclude=("tests",)),
    install_requires=requirements,
    python_requires=">=3",
)






