import gym
from stable_baselines3 import A2C, PPO, SAC
import os

env = gym.make('HalfCheetah-v3')

env.reset()

model_name = input('Select Model: \n A2C \n PPO \n SAC \n')

# Define storage directories
models_dir = f'models/{model_name}'
log_dir = 'logs'

if model_name == 'A2C':
    model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
elif model_name == 'PPO':
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
elif model_name == 'SAC':
    model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

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
