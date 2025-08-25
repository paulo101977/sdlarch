import os
import subprocess
import sys
import sysconfig

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

VERSION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sdlarch_rl/VERSION.txt",
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
README = open(os.path.join(SCRIPT_DIR, "README.md")).read()


setup(
    name="sdlarch-rl",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Paulo Sérgio",
    author_email="paulo10.1977@yahoo.com.br",
    url="https://github.com/paulo101977/sdlarch-rl",
    version=open(VERSION_PATH).read().strip(),
    license="MIT",
    install_requires=[
        "gymnasium>=0.27.1",
        "pyglet>=1.3.2,==1.*",
    ],
    python_requires=">=3.8.0,<3.13",
    packages=[
        "sdlarch_rl",
        "sdlarch_rl.utils",
        "sdlarch_rl.cores", 
        "sdlarch_rl.roms"
    ]
)