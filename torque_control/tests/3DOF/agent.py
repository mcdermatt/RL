from model import Actor, Critic #OG 2 layer
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from replayBuffer import ReplayBuffer
from OUNoise import OUNoise

device = torch.device("cuda:0")
# device = torch.device("cpu")


lrActor  = 0.001# 0.0001
lrCritic = 0.001 #0.0001
WEIGHT_DECAY =  0.001
BUFFER_SIZE = 1000000 #1000000 #uses last 10k trials (100 steps each)
BATCH_SIZE = 256 #2048 #128 #1024
discount_factor = 0.99 #0.99
TAU = 0.001 #0.005

class Agent():

	def __init__(self, state_size, action_size, LR_ACTOR = lrActor, LR_CRITIC = lrCritic, BATCH_SIZE = BATCH_SIZE, gamma = discount_factor):

		self.state_size = state_size
		self.action_size = action_size
		self.lr_critic = LR_CRITIC
		self.lr_actor = LR_ACTOR
		self.batch_size = BATCH_SIZE
		self.discount_factor = gamma

		self.actor = Actor(state_size,action_size).to(device)
		self.actor_target = Actor(state_size,action_size).to(device)
		self.critic = Critic(state_size,action_size).to(device)
		self.critic_target = Critic(state_size,action_size).to(device)

		#Initialize the target networks as copies of the original networks
		for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
			target_param.data.copy_(param.data)
		for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
			target_param.data.copy_(param.data)

		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = self.lr_actor)
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = self.lr_critic) #do I need weight decay?

		self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, self.batch_size)

		self.noise = OUNoise(self.action_size)

		self.aLossOut = 0
		self.cLossOut = 0

	def step(self, state, action,reward,next_state,done):
		"""save experience to memory buffer"""
		self.memory.add(state,action,reward,next_state,done)

		#sample (returns numpy array)
		if len(self.memory) > self.batch_size:
			experiences = self.memory.sample()
			self.learn(experiences, discount_factor)

	def learn(self, experiences, discount_factor):
		states, actions, rewards, next_states, dones = experiences

		#critic-------------------------------
		Qvals = self.critic(states,actions)
		next_actions = self.actor_target(next_states)
		next_Q = self.critic_target(next_states, next_actions)
		Qprime = rewards + discount_factor*next_Q #ignores result of samples that are at the end

		# closs = nn.SmoothL1Loss() #switched to this because I was getting negative actor loss (not possible)
		closs = nn.MSELoss() #most commonly used loss metric but error potentially explodes if there are outliars
		critic_loss = closs(Qvals,Qprime) #+ torch.rand(1) #+ 0.1*torch.rand(1) #ADD NOISE TO CRITIC
	
		self.cLossOut = critic_loss.cpu().detach().numpy()
	
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		# torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 4) #test- grad clipping
		self.critic_optimizer.step()

		# actor-------------------------------
		actions_pred = self.actor(states)
		actor_loss = -self.critic(states, actions_pred).mean()
		self.aLossOut = actor_loss.cpu().detach().numpy()

		# Minimize the loss
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		# torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 4) #test- grad clipping
		self.actor_optimizer.step()

		self.soft_update(self.critic, self.critic_target, TAU)
		self.soft_update(self.actor, self.actor_target, TAU)                     

	def soft_update(self, local_model, target_model, tau):
		"""θ_target = τ*θ_local + (1 - τ)*θ_target"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

	def save_models(self):
		# print('saving checkpoint')
		self.actor.save_checkpoint()
		self.critic.save_checkpoint()

	def load_models(self):
		self.actor.load_checkpoint()
		self.critic.load_checkpoint()