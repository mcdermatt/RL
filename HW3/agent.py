from model import Actor, Critic
from ragdoll import ragdoll
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from replayBuffer import ReplayBuffer


# device = torch.device("cuda:0")
device = torch.device("cpu")


LR_ACTOR = 0.0001
LR_CRITIC = 0.001
WEIGHT_DECAY = 0.001
BUFFER_SIZE = 1000000
BATCH_SIZE = 10
discount_factor = 0.99
TAU = 0.001

class Agent():

	def __init__(self, state_size, action_size):

		self.state_size = state_size
		self.action_size = action_size

		#init actor
		self.actor_local = Actor(state_size,action_size).to(device)
		self.actor_target = Actor(state_size,action_size).to(device)
		self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = LR_ACTOR)
		#init critic
		self.critic_local = Critic(state_size,action_size).to(device)
		self.critic_target = Critic(state_size,action_size).to(device)
		self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = LR_ACTOR, weight_decay = WEIGHT_DECAY)

		self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

	def step(self, state, action,reward,next_state,done):
		"""save experience to memory buffer"""
		self.memory.add(state,action,reward,next_state,done)

		if len(self.memory) > BATCH_SIZE:
			experiences = self.memory.sample()
			self.learn(experiences, discount_factor)

	# def act(self, state):

	# 	pass

	def learn(self, experiences, discount_factor):
		"""Update policy and value parameters using given batch of experience tuples.
		Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
		where:
		    actor_target(state) -> action
		    critic_target(state, action) -> Q-value
		Params
		======
		    experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
		    gamma (float): discount factor
		"""
		# print("learning")
		states, actions, rewards, next_states, dones = experiences

		# ---------------------------- update critic ---------------------------- #
		# Get predicted next-state actions and Q values from target models
		actions_next = self.actor_target(next_states)
		Q_targets_next = self.critic_target(next_states, actions_next)
		# Compute Q targets for current states (y_i)
		Q_targets = rewards + (discount_factor * Q_targets_next * (1 - dones))
		# Compute critic loss
		Q_expected = self.critic_local(states, actions)
		critic_loss = F.mse_loss(Q_expected, Q_targets)
		# Minimize the loss
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# ---------------------------- update actor ---------------------------- #
		# Compute actor loss
		actions_pred = self.actor_local(states)
		actor_loss = -self.critic_local(states, actions_pred).mean()
		# Minimize the loss
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# ----------------------- update target networks ----------------------- #
		self.soft_update(self.critic_local, self.critic_target, TAU)
		self.soft_update(self.actor_local, self.actor_target, TAU)                     

	def soft_update(self, local_model, target_model, tau):
		"""Soft update model parameters.
		θ_target = τ*θ_local + (1 - τ)*θ_target
		Params
		======
		    local_model: PyTorch model (weights will be copied from)
		    target_model: PyTorch model (weights will be copied to)
		    tau (float): interpolation parameter 
		"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
