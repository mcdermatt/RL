from model import Actor, Critic
from ragdoll import ragdoll
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from replayBuffer import ReplayBuffer
from OUNoise import OUNoise

device = torch.device("cuda:0")
# device = torch.device("cpu")


LR_ACTOR = 0.0001# 0.0001
LR_CRITIC = 0.001 #0.001
WEIGHT_DECAY = 0.001
BUFFER_SIZE = 100000 #1000000
BATCH_SIZE =  10 # was 10, tried 100 -> very slow on cpu
discount_factor = 0.99
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
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = LR_ACTOR, weight_decay = WEIGHT_DECAY)

		self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

		self.noise = OUNoise(action_size)

	def step(self, state, action,reward,next_state,done, is_learning = True):
		"""save experience to memory buffer"""
		#need to conver to numpy array
		self.memory.add(state,action,reward,next_state,done)
		# print(state,action,reward,next_state,done)

		#sample (returns numpy array)
		if len(self.memory) > BATCH_SIZE and is_learning == True:
			experiences = self.memory.sample()
			# print("learning")
			# print("experiences = ", experiences)
			self.learn(experiences, discount_factor)

	#TODO - figure out how to empty experiences/ memory?

	def learn(self, experiences, discount_factor):
		"""Update policy and value parameters using given batch of experience tuples.
		Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
		where:
		    actor_target(state) -> action
		    critic_target(state, action) -> Q-value

		    Targets are time delayed copies of original networks slowly track learned network
		Params
		======
		    experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
		    gamma (float): discount factor
		"""
		# print("learning")
		states, actions, rewards, next_states, dones = experiences

		# ---------------------------- update critic ---------------------------- #
		#Attempt 1:
		# # Get predicted next-state actions and Q values from target models
		# actions_next = self.actor_target(next_states)
		# Q_targets_next = self.critic_target(next_states, actions_next)

		# # Compute Q targets for current states (y_i)
		# Q_targets = rewards + (discount_factor * Q_targets_next * (1 - dones)) #if done, ignore result of next smaple
		# # print("Q_targets = ", Q_targets) # Q_targets = Q'

		# # Compute critic loss
		# Q_expected = self.critic(states, actions) #Qvals
		# # print("Q_expected = ", Q_expected)

		# # critic_loss = F.mse_loss(Q_expected, Q_targets) #mean squared error
		# # critic_loss = nn.MSELoss(Q_expected, Q_targets) #mean squared error


		# #switching from mse_loss
		# closs = nn.SmoothL1Loss()
		# critic_loss = closs(Q_expected, Q_targets) #aka Huber Loss


		# # print("critic_loss = ", critic_loss)
		# # Minimize the loss
		# self.critic_optimizer.zero_grad()
		# critic_loss.backward()
		# self.critic_optimizer.step()

		#Attempt 2:
		Qvals = self.critic.forward(states,actions)
		next_actions = self.actor_target.forward(next_states)
		next_Q = self.critic_target.forward(next_states, next_actions)
		Qprime = rewards + discount_factor*next_Q*(1-dones) #ignores result of samples that are at the end

		closs = nn.SmoothL1Loss()
		# closs = nn.MSELoss()
		critic_loss = closs(Qvals,Qprime)
		# print("critic_loss = ", critic_loss)
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# ---------------------------- update actor ---------------------------- #
		# Compute actor loss
		actions_pred = self.actor(states)
		actor_loss = -self.critic(states, actions_pred).mean()
		# actor_loss = self.critic(states, actions_pred).mean()
		# print("actor_loss = ", actor_loss)
		# Minimize the loss
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# ----------------------- update target networks ----------------------- #
		self.soft_update(self.critic, self.critic_target, TAU)
		self.soft_update(self.actor, self.actor_target, TAU)                     

	def soft_update(self, local_model, target_model, tau):
		"""Soft update model parameters.
		θ_target = τ*θ_local + (1 - τ)*θ_target
		Params
		======
		    local_model: PyTorch model (weights will be copied from)
		    target_model: PyTorch model (weights will be copied to)
		    tau (float): interpolation parameter 

		Brings Parameters from target network back to training network
		"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
