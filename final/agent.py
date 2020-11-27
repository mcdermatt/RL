from model import Actor, Critic
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from replayBuffer import ReplayBuffer
from OUNoise import OUNoise

# device = torch.device("cuda:0")
device = torch.device("cpu")


LR_ACTOR = 0.0001# 0.0001
LR_CRITIC = 0.001 #0.001
WEIGHT_DECAY = 0.001
BUFFER_SIZE = 100000
BATCH_SIZE =  20 #TODO - figure out if I need this for instant rewards- I think I do
discount_factor = 0.99 #TODO - figure out if I need this - I think I do not
TAU = 0.001

class Agent():

	def __init__(self, state_size, action_size):

		self.state_size = state_size
		self.action_size = action_size

		#init actor
		self.actor = Actor(state_size,action_size).to(device)
		self.actor_target = Actor(state_size,action_size).to(device)
		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = LR_ACTOR)
		#init critic
		self.critic = Critic(state_size,action_size).to(device)
		self.critic_target = Critic(state_size,action_size).to(device)
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = LR_CRITIC, weight_decay = WEIGHT_DECAY)

		self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

		self.noise = OUNoise(action_size)

		self.aLossOut = 0
		self.cLossOut = 0

	def step(self, state, action,reward,next_state,done, is_learning = True):
		"""save experience to memory buffer"""
		self.memory.add(state,action,reward,next_state,done)

		#sample (returns numpy array)
		if len(self.memory) > BATCH_SIZE and is_learning == True:
			experiences = self.memory.sample()
			self.learn(experiences, discount_factor)

	def learn(self, experiences, discount_factor):
		states, actions, rewards, next_states, dones = experiences

		#update critic
		Qvals = self.critic(states,actions)
		# next_actions = self.actor_target(next_states)
		# next_Q = self.critic_target(next_states, next_actions)

		closs = nn.SmoothL1Loss()
		# closs = nn.MSELoss() - error too large, this explodes
		critic_loss = closs(Qvals,rewards)
		self.cLossOut = critic_loss.cpu().detach().numpy()
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss
		actions_pred = self.actor(states)
		actor_loss = -self.critic(states, actions_pred).mean()
		self.aLossOut = actor_loss.cpu().detach().numpy()

		# Minimize the loss
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		self.soft_update(self.critic, self.critic_target, TAU)
		self.soft_update(self.actor, self.actor_target, TAU)                     

	def soft_update(self, local_model, target_model, tau):
		"""θ_target = τ*θ_local + (1 - τ)*θ_target"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
