import gym
from stable_baselines3 import PPO, A2C

env = gym.make('HalfCheetah-v3')

env.reset()

mod = int(input('Select Model: \n 1 - PPO \n 2 - A2C \n'))

if mod == 1:
    model = A2C('MlpPolicy', env, verbose=1)
elif mod == 2:
    model = PPO('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=10000)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:

        env.render()
        obs, rewards, done, info = env.step(env.action_space.sample())

env.close()
