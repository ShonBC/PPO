import gym
from stable_baselines3 import A2C, PPO, SAC
import os

env = gym.make('HalfCheetah-v3')

env.reset()

mod = int(input('Select Model: \n 1 - A2C \n 2 - PPO \n 3 - SAC \n'))

log_dir = 'logs'

if mod == 1:
    model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
    model_name = 'A2C'
elif mod == 2:
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
    model_name = 'PPO'
elif mod == 3:
    model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
    model_name = 'SAC'

# Define storage directories
models_dir = f'models/{model_name}'


if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# How frequently model is saved
time_steps = 10000

for i in range(1, 30):  # Can be while True and stop when performance peaks
    model.learn(total_timesteps=time_steps, reset_num_timesteps=False,
                tb_log_name=model_name)

    model.save(f'{models_dir}/{time_steps*i}')

env.close()
