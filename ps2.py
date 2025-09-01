import ctypes
import numpy as np
import time
import cv2
import gc
import os
import zlib
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from sdlarch_rl.sdlenv import SDLEnv


# env = PCSX2Core("GranTurismo3-Ps2")

# obs, info = env.reset()

if __name__ == "__main__":
    def make_env():
        def _init():
            env = SDLEnv("GranTurismo3-Ps2", render_mode="rgb_array")
            return env
        return _init

    env = make_vec_env(make_env(), n_envs=2, vec_env_cls=SubprocVecEnv)

    model = PPO('CnnPolicy', 
    # model = RecurrentPPO('CnnLstmPolicy',
        env, 
        verbose=1, 
        # policy_kwargs=policy_kwargs, 
    )



    model.learn(total_timesteps=10_000, reset_num_timesteps=False)
    model.save("test")


# count = 0
# global initial_state
# initial_state = None

# while True:
#     action = np.zeros(16, dtype=np.uint8)

#     max_count = 3000
#     # press Y button

#     if count == 300:
#         if not initial_state:
#             print("Before get_state")
#             initial_state = env.unwrapped.em.get_state()
#             print("Before set_state")
#             time.sleep(2)
#             env.unwrapped.em.run()
#             env.unwrapped.em.set_state(initial_state)
#             print("After set_state")
    
#     if count % 100 == 0 and count > 0 and count < max_count:
#         # press start
#         if count < 1000:
#             action[3] = 1
#         action[0] = 1
#     elif count > max_count:
#         action = np.zeros(16, dtype=np.uint8)

#         action[0] = 1

#         if not initial_state:
#             initial_state = env.unwrapped.em.get_state()
            
#         if count > 4000:
#             env.unwrapped.em.run()
#             env.unwrapped.em.set_state(initial_state)
#             print("After set_state")
#             count = 3001

#     img, rew, done, _, info = env.step(action)
#     time.sleep(0.016)

#     # print(env.get_memory(0x01FA1E7C))
    
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     cv2.imshow("env", img)
#     cv2.waitKey(1)

#     count += 1

#     # break

#     if count % 1000 == 0:
#         # env.reset()
#         pass