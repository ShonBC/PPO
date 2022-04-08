import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class PPOMemory:
    def __init__(self, batch_size_):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size_

    def generate_batches(self):
        n_states = len(self.states)
        batch_start_indexes = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start_indexes]
        return np.array(self.states),\
            np.array(self.actions),\
            np.array(self.probs),\
            np.array(self.vals),\
            np.array(self.rewards),\
            np.array(self.dones),\
            batches

    def store_memory(self, state, action, prob, val, reward, done):

        self.states.append(state)
        self.action.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):

        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []


class ActorNetwok(nn.Module):
    pass


class CriticNetwork(nn.Module):

    def __init__(self, input_dims, alpha, chkpt_dir_):
        self.chkpt_dir = chkpt_dir_
        super().__init__()
        self.checkpoint_file = os.path.join(self.chkpt_dir, 'critic_ppo')
        # print(*input_dims)
        # print(input_dims)
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, 256),  # Not sure why there is an asterisk here!
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.critic(state)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent():

    def __init__(self, n_actions, input_dims, gamma, alpha, gae_lambda,
                 policy_clip, batch_size, n_epochs):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        # self.actor = ActorNetwok(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha, "models/")
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, prob, val, reward, done):
        self.memory.store_memory(state, action, prob, val, reward, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_aciton(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        actor_output = self.actor(state)
        value = self.critic(state)
        action = actor_output.sample()
        prob = T.squeeze(actor_output.log_prob(action)).item()  # Single value
        action = T.squeeze(action).item()  # Single vlaue
        value = T.squeeze(value).item()  # Single vlaue
        return action, prob, value

    def get_advantage(self, values, rewards, dones):
        advantage = np.zeros(len(rewards), dtype=np.float32)
        for t in range(len(rewards)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards)-1):
                a_t += discount*(rewards[k] + self.gamma*values[k+1] *
                                 (1-int(dones[k])) - values[k])
                discount *= self.gamma*self.gae_lambda
            advantage[t] = a_t

        return T.tensor(advantage).to(self.actor.device)

    def learn_batch(self, batch, states, actions, old_probs, values,
                    advantage):
        pass

    def learn(self):
        for _ in range(self.n_epochs):
            states, actions, old_probs, vals,\
                rewards, dones, batches = self.memory.generate_batches()
            advantage = self.get_advantage(vals, rewards, dones,)
            values = T.tensor(values).to(self.actor.device)

            for batch in batches:
                self.learn_batch(batch, states, actions, old_probs, values,
                                 advantage)

        self.memory.clear_memory()

