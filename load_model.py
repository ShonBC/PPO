import gym
from stable_baselines3 import A2C, PPO, SAC
import os

env = gym.make('HalfCheetah-v3')

env.reset()

model_name = input('Select Model: \n A2C \n PPO \n SAC \n')
epoch = input('Specify the desired epoch to load: ')
# Define storage directories
models_dir = f'models/{model_name}'
model_path = f'{models_dir}/{epoch}'
log_dir = 'logs'

if model_name == 'A2C':
    model = A2C.load(model_path, env=env)
elif model_name == 'PPO':
    model = PPO.load(model_path, env=env)
elif model_name == 'SAC':
    model = SAC.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:

        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

env.close()
