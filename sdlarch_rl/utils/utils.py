import gymnasium as gym
import cv2
import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import os
import re
import random

from pettingzoo import AECEnv
from gymnasium import spaces
import torch
from torch.utils.data import Dataset
# from imitation.data.types import Trajectory
# from imitation.data import rollout
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import torch
from gymnasium.spaces import MultiBinary, Discrete

class NormalizeObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )

    def observation(self, obs):
        return obs.astype(np.float32) / 255.0

class ExcludeButtonsWrapper(gym.ActionWrapper):
    def __init__(self, env, button_names, excluded_buttons):
        super().__init__(env)

        self.original_action_space = env.action_space
        self.button_names = button_names
        self.excluded_buttons = excluded_buttons
        self.filtered_actions = None

        if isinstance(env.action_space, MultiBinary):
            self.excluded_indices = [self._find_button_index(b) for b in excluded_buttons]

            if any(idx is None for idx in self.excluded_indices):
                raise ValueError("Todos os botões excluídos devem estar em button_names.")

        elif isinstance(env.action_space, Discrete):
            if not hasattr(env, 'combo_array'):
                raise AttributeError("Discrete space must have a combo_array attribute.")
            self._filter_combo_array()
            self.action_space = Discrete(len(self.filtered_actions))

        else:
            raise NotImplementedError("Only MultiBinary and Discrete action spaces are supported.")

    def _find_button_index(self, name):
        try:
            return self.button_names.index(name)
        except ValueError:
            return None

    def _filter_combo_array(self):
        def is_valid(combo):
            return not any(b in combo for b in self.excluded_buttons)

        self.filtered_actions = [a for a in self.env.combo_array if is_valid(a)]

    def action(self, action):
        if isinstance(self.original_action_space, MultiBinary):
            for idx in self.excluded_indices:
                action[idx] = 0
            return action

        elif isinstance(self.original_action_space, Discrete):
            return self.filtered_actions[action]

    def reverse_action(self, action):
        return action

augmentation_fn = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomApply([
        transforms.ColorJitter(0.1, 0.1),
        transforms.GaussianBlur(3)
    ], p=0.5),
    transforms.ToTensor()
])

def random_brightness(frame):
    factor = np.random.uniform(0.85, 1.15)
    frame = np.clip(frame * factor, 0, 255).astype(np.uint8)
    return frame

def add_noise(frame):
    noise = np.random.normal(0, 5, frame.shape).astype(np.uint8)
    frame = np.clip(frame + noise, 0, 255)
    return frame

def apply_blur(frame):
    # Apply Gaussian Blur with kernel 3x3 or 5x5
    if np.random.rand() < 0.5:
        return cv2.GaussianBlur(frame, (3, 3), 0)
    else:
        return frame

def random_shift(frame):
    max_shift = 1
    tx = np.random.randint(-max_shift, max_shift + 1)
    ty = np.random.randint(-max_shift, max_shift + 1)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REFLECT)
    return frame

def augment_trajectory(traj):

    new_obs = []
    for obs in traj.obs:
        frame = obs.copy()
        frame = random_brightness(frame)
        frame = apply_blur(frame)
        frame = random_shift(frame)
        new_obs.append(frame)
    
    new_traj = Trajectory(
        obs=np.stack(new_obs),
        acts=np.array(traj.acts),
        infos=traj.infos,
        terminal=traj.terminal
    )


    return new_traj

class RetroPettingZoo(AECEnv):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.players = ['player_0', 'player_1']
        self.agent_order = self.players[:]
        self.possible_agents = self.players[:]
        self.agents = self.players[:]

        obs_space = self.env.observation_space
        self.observation_spaces = {agent: obs_space for agent in self.players}
        self.action_spaces = {agent: spaces.Discrete(12) for agent in self.players}

        self._agent_selector = self._agent_selector_func()
        self.last_observation = None
        self.last_rewards = {agent: 0 for agent in self.players}

    def _agent_selector_func(self):
        while True:
            for agent in self.agent_order:
                yield agent

    def reset(self, seed=None, options=None):
        self.agents = self.players[:]
        self._agent_selector = self._agent_selector_func()
        self.current_agent = next(self._agent_selector)
        obs, _ = self.env.reset()
        self.last_observation = obs
        return {agent: obs for agent in self.agents}, {}

    def observe(self, agent):
        return self.last_observation

    def step(self, action):
        if self.dones[self.current_agent]:
            self._was_done_step(action)
            return

        if self.current_agent == 'player_0':
            self._last_p1_action = action
            self.current_agent = next(self._agent_selector)
        else:
            self._last_p2_action = action
            full_action = [self._last_p1_action, self._last_p2_action]
            obs, rewards, term, _, _ = self.env.step(full_action)
            self.last_observation = obs

            self.rewards = {
                "player_0": rewards[0],
                "player_1": rewards[1]
            }
            self.dones = {
                "player_0": term,
                "player_1": term
            }
            self.current_agent = next(self._agent_selector)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

class AugmentObservation(gym.ObservationWrapper):
    def __init__(self, env, noise=False, debug=False):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.debug=debug
        self.noise=noise

    def observation(self, obs):
        # obs: shape (stack, H, W, C)
        augmented = obs

        if np.random.rand() < 0.5:
            augmented = np.zeros_like(obs)
    
            # for i in range(obs.shape[0]):
            frame = obs.copy()
            frame = self.random_brightness(frame)
            frame = self.safe_blur(frame)
            if self.noise:
                frame = self.add_noise(frame)
            frame = self.random_shift(frame)
                
            # augmented[i] = frame
            augmented = frame
    
            if self.debug:
                obs_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                cv2.imshow("Augmentation", obs_bgr)
                cv2.waitKey(1)
            return augmented

        return augmented

    def random_brightness(self, frame):
        factor = np.random.uniform(0.85, 1.15)
        frame = np.clip(frame * factor, 0, 255).astype(np.uint8)
        return frame

    def add_noise(self, frame):
        noise = np.random.normal(0, 5, frame.shape).astype(np.uint8)
        frame = np.clip(frame + noise, 0, 255)
        return frame

    def safe_blur(self, frame):
        """Blur applied to bck, not to ememy/missiles"""
        if np.random.rand() < 0.4:
            important_mask = self.get_important_objects_mask(frame)
            
            blurred = cv2.GaussianBlur(frame, (3, 3), 0)
            
            result = np.where(important_mask[..., np.newaxis], frame, blurred)
            return result.astype(np.uint8)
        return frame

    def get_important_objects_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        red_low = np.array([0, 120, 70])
        red_high = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, red_low, red_high)
        
        blue_low = np.array([90, 50, 50])
        blue_high = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_low, blue_high)
        
        combined = np.logical_or(red_mask > 0, blue_mask > 0)
        
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(combined.astype(np.uint8), kernel, iterations=1)

    def random_shift(self, frame):
        max_shift = 1
        tx = np.random.randint(-max_shift, max_shift + 1)
        ty = np.random.randint(-max_shift, max_shift + 1)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REFLECT)
        return frame

class RandomStateWrapper(gym.Wrapper):
    def __init__(self, env, states):
        super().__init__(env)
        
        self.env = env
        self.states = states

    """Select a random state from folder STATE_PATH"""
    def reset(self, **kwargs):
        self.env.load_state(random.choice(self.states))
        
        obs, info = self.env.reset(**kwargs)
    
        return  obs, info

class HPInfoWrapper(gym.Wrapper):
    def __init__(self, env, start_x, end_x, start_y, end_y, max_hp_pixels, start_color, end_color, debug = False):
        super().__init__(env)
        self.max_hp_pixels = max_hp_pixels
        self.start_x = start_x
        self.end_x = end_x
        self.start_y = start_y
        self.end_y = end_y
        self.start_color = start_color
        self.end_color = end_color
        self.debug = debug

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Extract the HP bar
        hp_roi = obs[self.start_x:self.end_x, self.start_y:self.end_y].copy()

        # RGB to HSV
        hsv = cv2.cvtColor(hp_roi, cv2.COLOR_RGB2HSV)

        # mask to color
        lower = np.array([self.start_color, 100, 100])
        upper = np.array([self.end_color, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        if self.debug:
            hp_roi_db = cv2.resize(hp_roi, ((self.end_y - self.start_y) * 3, (self.end_x - self.start_x) * 3), interpolation=cv2.INTER_AREA)
            cv2.imshow("HP Bar", hp_roi_db)
            # cv2.imshow("Mask Bar", mask)
            cv2.waitKey(1)


        # Count the pixels and calculate the HP percentual
        hp_pixels = cv2.countNonZero(mask)
        hp_percent = hp_pixels / self.max_hp_pixels

        # Inject on dictionary info
        info["hp_percent"] = hp_percent
        info["hp_pixels"] = hp_pixels

        return obs, reward, terminated, truncated, info

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            observation, reward, terminated, trunk, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        return observation, total_reward, terminated, trunk, info

def get_latest_model(path):
    models = list(path.glob("best_model_*"))
    if not models:
        return None
    model_numbers = [int(re.search(r"best_model_(\d+)", str(m)).group(1)) for m in models]
    latest_model = max(model_numbers)
    return path / f"best_model_{latest_model}"

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.shape[1], self.shape[0], 3), dtype=np.uint8
        )

    def observation(self, obs):
        return cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)


class GrayResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, keep_dim=True):
        super().__init__(env)
        self.width = width
        self.height = height
        self.keep_dim = keep_dim  # True → (H, W, 1), False → (H, W)

        shape = (self.height, self.width, 1) if keep_dim else (self.height, self.width)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

    def observation(self, obs):
        # obs: RGB image with shape (H, W, 3)
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)

        if self.keep_dim:
            resized = np.expand_dims(resized, axis=-1)  # (H, W, 1)

        return resized.astype(np.uint8)


class TrainAndLoggingCallback(BaseCallback):
    def __init__(
            self, 
            check_freq, 
            save_path, 
            save_freq, 
            model, 
            use_curriculum = False, 
            verbose=1, 
            reward_net=None, 
            logger=None,
            use_call=False
        ):
        
        super(TrainAndLoggingCallback, self).__init__(verbose)
        
        self.check_freq = check_freq
        self.save_freq = save_freq
        self.save_path = save_path
        self.use_curriculum = use_curriculum
        self.model = model
        self.reward_net=reward_net

        self.episode_rewards = []
        self.current_episode_reward = 0
        self.counter = 0
        self.my_logger =  None
        self.use_call = use_call

        if logger is not None:
            self.my_logger = logger

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def __call__(self, _locals=None, _globals=None):
        if self.use_call:
            self.counter += 1
    
            latest_model = get_latest_model(self.save_path)
            next_save_step = (int(re.search(r"best_model_(\d+)", str(latest_model)).group(1)) + 1) if latest_model else self.counter 
            model_path = self.save_path / f"best_model_{next_save_step}"
            reward_path = self.save_path / f"reward_net_{next_save_step}.pt"
            self.model.save(model_path)
            torch.save(self.reward_net.state_dict(), reward_path)
    
            
            # logger = self.model.get_logger()
            for key, value in self.logger.name_to_value.items():
                self.my_logger.record(key, value)
    
            next_save_step = int(next_save_step)
    
            # self.my_logger.record("custom/timestep", next_save_step)
            self.my_logger.dump(next_save_step)
    
            print("timestep", next_save_step)
            
            print(f"Model saved in: {model_path}")

    def _on_step(self):
        reward = self.locals["rewards"][0]
        self.current_episode_reward = reward

        done = self.locals["dones"][0]

        self.episode_rewards.append(self.current_episode_reward)

        if done:
            steps_count = len(self.episode_rewards)

            if self.use_curriculum:
                self.logger.record("current_phase", self.training_env.get_attr("current_phase")[0])

            print(f"Done Rewards Step Cnt: {steps_count}")
            self.episode_rewards = []
        
        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) > 0:
            latest_model = get_latest_model(self.save_path)
            next_save_step = (int(re.search(r"best_model_(\d+)", str(latest_model)).group(1)) + self.check_freq) if latest_model else self.n_calls
            model_path = self.save_path / f"best_model_{next_save_step}"
            self.model.save(model_path)
            print(f"Model saved in: {model_path}")

        
        return True

class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env, required_wins=20, required_avg_reward=1.0):
        super().__init__(env)
        self.required_wins = required_wins
        self.required_avg_reward = required_avg_reward
        self.current_phase = 1
        self.total_wins = 0 
        self.rewards_list = []

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        return obs

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        self.rewards_list.append(reward)

        could_to_next_stage = info["matches_won"] / self.current_phase >= 2

        if info["matches_won"] % 2 == 0 and info["matches_won"] > 0 and could_to_next_stage:
            self.total_wins += 1


        avg_reward = np.mean(self.rewards_list[-self.required_wins:]) if len(self.rewards_list) >= self.required_wins else np.mean(self.rewards_list)

        if could_to_next_stage and \
            ((info["matches_won"] % 2 == 0  and info["matches_won"] > 0) \
                 or (info["enemy_matches_won"] % 2 == 0 and info["enemy_matches_won"] > 0)) :
            print(info)
            print(f"🔥 stage {self.current_phase}! ({self.total_wins} fights win, avg rewards: {avg_reward:.2f})")
            done = True
        
        if self.total_wins >= self.required_wins and avg_reward >= self.required_avg_reward:
            self.current_phase += 1
            print(f"🔥 Going to next stage {self.current_phase}! ({self.total_wins} fights win, avg rewards: {avg_reward:.2f})")
            self.total_wins = 0
            self.rewards_list = []

        return obs, reward, done, truncated, info