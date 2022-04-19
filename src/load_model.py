import gym
from stable_baselines3 import A2C, PPO, SAC
import os
import torch

from ppo import Agent

env = gym.make('HalfCheetah-v3')

env.reset()

model_name = input('Select Model: \n A2C \n PPO \n SAC \n CustomPPO \n')
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
    print(model)
elif model_name == 'CustomPPO':
    batch_size = 5
    alpha = 0.0003
    n_epochs = 4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Agent(n_actions=env.action_space.shape[0],
                  batch_size=batch_size,
                  alpha=alpha,
                  n_epochs=n_epochs,
                  input_dims=env.observation_space.shape,
                  gamma=0.99,
                  gae_lambda=0.95,
                  policy_clip=0.2
                  )
    model.load_models()

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:

        env.render()
        if model_name == 'CustomPPO':
            action, prob, val = model.choose_action(obs)
        else:
            action, _ = model.predict(obs)

        obs, reward, done, info = env.step(action)

env.close()
