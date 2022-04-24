import gym
import numpy as np
import matplotlib.pyplot as plt
from ppo import Agent
from torch.utils.tensorboard import SummaryWriter
import os

environment = 'HalfCheetah-v3'  # 'CartPole-v0'
env = gym.make(environment)

time_steps = 10000
batch_size = 64
n_epochs = 10
alpha = 0.0003
n_games = 300
learn_iters = 0
avg_score = 0
n_steps = 0
best_score = env.reward_range[0]
score_history = []
figure_dir = 'plots/CustomPPO/'
figure_file = figure_dir + 'games_' + str(n_games) + ',epochs_' + \
            str(n_epochs) + ',alpha_' + str(alpha) + '.png'
print(figure_file)
print(env.action_space)
print(env.observation_space.shape)

log_dir = 'logs/CustomPPO'
chkpt_dir = "models/CustomPPO/"
writer = SummaryWriter(log_dir)

if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(chkpt_dir):
    os.makedirs(chkpt_dir)

agent = Agent(n_actions=env.action_space.shape[0],
              batch_size=batch_size,
              alpha=alpha,
              n_epochs=n_epochs,
              input_dims=env.observation_space.shape,
              gamma=0.99,
              gae_lambda=0.95,
              policy_clip=0.2,
              chkpt_dir="models/CustomPPO/"
              )


for i in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action, prob, val = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        n_steps += 1
        score += reward
        agent.remember(observation, action, prob, val, reward, done)

        if n_steps % time_steps == 0 or n_steps == 1000:
            agent.learn(n_steps)
            learn_iters += 1

        observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    writer.add_scalar("rollout/ep_rew_mean", score, n_steps)
    writer.flush()

    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()

    print('Episode', i, ',',
          'score %.1f' % score, ',',
          'avg score %.1f' % avg_score, ',',
          'time_steps', n_steps, ',',
          'learning_steps', learn_iters
          )

x = [i + 1 for i in range(len(score_history))]
plt.plot(x, score_history)
plt.savefig(figure_file)
plt.show()
# writer.flush()
