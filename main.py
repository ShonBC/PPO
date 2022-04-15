import gym
import numpy as np
import matplotlib.pyplot as plt
from ppo import Agent

environment = 'HalfCheetah-v2'#'CartPole-v0'  
env = gym.make(environment)

N = 20
batch_size = 5
n_epochs = 4
alpha = 0.0003
n_games = 300
learn_iters = 0
avg_score = 0
n_steps = 0
best_score = env.reward_range[0]
score_history = []
figure_file = 'plots/CustomPPO/' + 'games=' + str(n_games) + ',epochs='+ str(n_epochs) + ',alpha='+  str(alpha) + '.png'
print(figure_file)


agent = Agent(n_actions = env.action_space.n, 
              batch_size=batch_size,
              alpha = alpha, 
              n_epochs=n_epochs, 
              input_dims=env.observation_space.shape,
              gamma = 0.99,
              gae_lambda = 0.95,
              policy_clip = 0.2
              )


for i in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action, prob, val = agent.choose_aciton(observation)
        observation_, reward, done, info = env.step(action)
        n_steps += 1
        score += reward
        agent.remember(observation, action, prob, val, reward, done)
        
        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1
            
        observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    
    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()
    
    print('Episode', i, ',',
          'score %.1f' %score, ',',
          'avg score %.1f' %avg_score, ',',
          'time_steps', n_steps, ',',
          'learning_steps', learn_iters       
          )

x = [i + 1 for i in range(len(score_history))]
plt.plot(x, score_history)
plt.savefig(figure_file)
plt.show()

