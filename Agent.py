import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from Model import GaussianPolicy, QNetwork, DeterministicPolicy
from collections import deque
import copy
import random
import numpy as np



class SAC(object):
    def __init__(self, input_dim, action_space, hidden_dim, batch_size, buffer_size):

        self.alpha = 0.01
        self.lr = 5e-4
        self.device = torch.device("cuda")

        self.critic = QNetwork(input_dim, action_space.shape[0], hidden_dim).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.policy = GaussianPolicy(input_dim, action_space.shape[0], hidden_dim, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0].round(2)

    def store(self, data):
        state, action, reward = data
        self.buffer.append(copy.deepcopy([state, action, reward]))

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return 0, 0, 0
        else:
            batch = random.sample(self.buffer, self.batch_size)
            states, actions, rewards = zip(*batch)
            batch = (np.array(states), np.array(actions), np.array(rewards))
            return self.update_parameters(batch)

    def update_parameters(self, batch):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch = batch

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)

        next_q_value = reward_batch.detach()
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # J娄脨 = ??st隆芦D,娄脜t隆芦N[娄脕 * log娄脨(f(娄脜t;st)|st) ? Q(st,f(娄脜t;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            if evaluate:
                self.policy.eval()
                self.critic.eval()
            else:
                self.policy.train()
                self.critic.train()