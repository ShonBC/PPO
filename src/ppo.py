import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


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
        self.actions.append(action)
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
    def __init__(self, n_actions, input_dims,
                 alpha, chkpt_dir,
                 fc1_dims=256, fc2_dims=256, ):
        super().__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo.pth')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):

    def __init__(self, input_dims, alpha, chkpt_dir_):
        self.chkpt_dir = chkpt_dir_
        super().__init__()
        self.checkpoint_file = os.path.join(self.chkpt_dir, 'critic_ppo.pth')
        # print(*input_dims)
        # print(input_dims)
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, 256),
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
                 policy_clip, batch_size, n_epochs, chkpt_dir):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwok(n_actions, input_dims, alpha,
                                 chkpt_dir="models/CustomPPO/")
        self.critic = CriticNetwork(input_dims, alpha, "models/CustomPPO/")
        self.memory = PPOMemory(batch_size)
        self.writer = SummaryWriter(log_dir='logs/CustomPPO')

    def remember(self, state, action, prob, val, reward, done):
        self.memory.store_memory(state, action, prob, val, reward, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor(np.array(observation),
                         dtype=T.float).to(self.actor.device)
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

    def learn_batch(self, batch, states_arr, actions_arr, old_probs_arr,
                    values, advantage):

        states = T.tensor(states_arr[batch],
                          dtype=T.float).to(self.actor.device)

        old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
        actions = T.tensor(actions_arr[batch]).to(self.actor.device)

        dist = self.actor(states)
        critic_value = self.critic(states)

        critic_value = T.squeeze(critic_value)

        new_probs = dist.log_prob(actions)
        prob_ratio = new_probs.exp() / old_probs.exp()

        weighted_probs = advantage[batch] * prob_ratio
        weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                         1+self.policy_clip) * advantage[batch]

        actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

        returns = advantage[batch] + values[batch]
        critic_loss = (returns - critic_value)**2
        critic_loss = critic_loss.mean()

        total_loss = actor_loss + 0.5 * critic_loss
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        total_loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()
        return prob_ratio, actor_loss, returns, critic_loss, total_loss

    def learn(self, n_steps):
        for i in range(self.n_epochs):
            states, actions, old_probs, vals,\
                rewards, dones, batches = self.memory.generate_batches()
            advantage = self.get_advantage(vals, rewards, dones,)
            values = T.tensor(vals).to(self.actor.device)

            for batch in batches:
                prob_ratio, actor_loss, returns, critic_loss, total_loss = self.learn_batch(batch, states, actions, old_probs, values, advantage)

        self.writer.add_scalar("train/clip_fraction", prob_ratio.mean(), n_steps)
        self.writer.add_scalar("train/loss", total_loss, n_steps)
        self.writer.add_scalar("train/actor_loss", actor_loss, n_steps)
        self.writer.add_scalar("train/critic_loss", critic_loss, n_steps)

        self.memory.clear_memory()
