import os
import sys
from sdlarch_rl.sdlenv import SDLEnv
from sdlarch_rl.utils.discretizer import MainDiscretizer

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(os.path.dirname(__file__), "VERSION.txt")) as f:
    __version__ = f.read()


__all__ = [
    "SDLEnv",
    "MainDiscretizer",
    "make",
]


def make(game,  **kwargs):
    """
    Create a Gym environment for the specified game
    """
    return SDLEnv(game, **kwargs)


