
import os
import time
import numpy as np
import gymnasium as gym
import json
import cv2
import importlib.util as import_util
import gc
from _retro import RetroEmulator
import ctypes

class PCSX2Core(gym.Env):
    """
    PCSX2 environment class

    Provides a Gym interface to classic video games
    """

    metadata = {"render_modes": ["human", "rgb_array"], "video.frames_per_second": 60.0}

    def __init__(
        self, 
        gamename: str,
        players = 1,
        env_id = 0,
    ) -> None:

        self.em = RetroEmulator()
        self.players = players

        # TODO: get correct path to core and game
        core = os.path.abspath("../cores/pcsx2_libretro.so")
        game = os.path.abspath("../../roms/Granturismo3.iso")
        
        gc.collect()

        if not hasattr(self, "spec"):
            self.spec = None

        self.dirname = os.path.dirname(__file__)
        

        if not os.path.exists(os.path.join(self.dirname, r"roms", f"{gamename}")):
            raise FileNotFoundError(
                f"Game directory not found: {os.path.join(self.dirname, r'roms', f'{gamename}')}. Please ensure the path is correct."
            )

        game = os.path.join(self.dirname, r"roms", f"{gamename}", r"rom.iso")

        if not os.path.isfile(game):
            raise FileNotFoundError(f"ROM file not found: {rom}. Please ensure the path is correct.")

        # rum the emulator main process
        self.em.init(core, game)

        self.em.run()

        pcsx2_json = os.path.join(self.dirname, r"cores/ps2/pcsx2.json")

        with open(pcsx2_json) as f:
            pcsx2_button = json.load(f)

        self.buttons = pcsx2_button['buttons']


        meta_path = os.path.join(self.dirname, r"roms", f"{gamename}", f"meta.json")

        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Meta file not found: {meta_path}. Please ensure the path is correct.")

        with open(meta_path) as meta:
            self.meta = json.load(meta)

        self.action_space = gym.spaces.MultiBinary(len(self.buttons) * players)

        observation = self._get_observation()

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=observation.shape,
            dtype=np.uint8,
        )
        
        self.img = None

        # TODO: load reward code dynamically
        reward_path = os.path.join(self.dirname, r"roms", f"{gamename}", f"reward.py")

        if not os.path.isfile(reward_path):
            raise FileNotFoundError(f"Reward file not found: {reward_path}. Please ensure the path is correct.")

        # Load the reward function from the specified file
        spec = import_util.spec_from_file_location("dynamic_module", reward_path)
        module = import_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.reward_fn = module.reward

        self.count = 0

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """
        Reset the controller and ensure the PCSX2 emulator is started.
        :return: A tuple containing the next state and additional info.
        """

        super().reset(seed=seed, options=options)

        self.em.run()

        observation = self._get_observation()

        # TODO: get initial info from memory
        # TODO: reset to a specific state

        self.count = 0
        
        return observation, {}

    def get_memory(self, address: int) -> int:
        """
        Read a byte from the specified memory address.
        """

        data = self.em.get_ram()[address]
        return data

    def step(self, actions: np.ndarray):
        """
        Execute one time step within the environment.
        :param actions: The actions to be executed.
        :return: A tuple containing the next state, reward, done flag, truncated, and additional info.
        """

        if self.img is None:
            raise RuntimeError("Please call env.reset() before env.step()")

        # TODO: set buttons for all players
        for player in range(self.players):
            self.em.set_button_mask(actions, player)

        self.em.run()

        observation = self._get_observation()

        # TODO: get info from memory
        # info = self._memory_to_info()

        # TODO: get reward from info
        # reward, done = self._get_reward(self.old_info, info)

        info = {}
        self.old_info = info

        self.count += 1

        # TODO: set reward, done, truncated, info correctly
        return observation, 0, False, False, {}

    def close(self) -> None:
        """
        Close the controller and clean up resources.
        """
        # TODO: properly close the emulator
        # if self.process:
        #     self.process.terminate()
        #     self.process.wait()

        # if hasattr(self, 'libipc'):
        #     self.libipc.pine_pcsx2_delete(self.ipc)
        pass

    def _get_observation(self) -> np.ndarray:
        height, width = self.em.get_shape()

        buffer = (ctypes.c_uint8 * (width * height * 3))()
        self.em.get_frame(buffer, width, height)
        
        img = np.frombuffer(buffer, dtype=np.uint8)
        img = img.reshape((height, width, 3))[::-1]
        self.img = img
        return self.img

    
    def _memory_to_info(self) -> dict:
        """
        Reads specific memory addresses to extract game-related information.
        :return: A dictionary containing game-related information.
        """

        info = {
        }

       
        return info

    def _get_reward(self, old_info: dict, info: dict) -> tuple[dict, dict]:
        """
        Calculate the reward based on the current game state.
        :param      old_info: The previous state information.
        :param      info: The current state information.
        :return:   A tuple containing the reward and a boolean indicating if the episode is done.
        """
        return self.reward_fn(old_info, info)
        