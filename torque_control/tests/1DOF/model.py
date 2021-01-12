import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

WFINAL = 0.003

class Actor(nn.Module): #create actor class and inherit from nn.Module
	def __init__(self, state_size = 6, action_size = 9, nodes1 = 16, nodes2 = 16): #was 400, 300
		super(Actor,self).__init__() #need to run this because init func of nn.Module is not run upon inherit
		self.checkpoint_file = "checkpoint/actor"

		#Linear is a simple flat fuly connected
		self.fc1 = nn.Linear(state_size, nodes1) #input current 13 state observations
		self.fc2 = nn.Linear(nodes1, nodes2)  #arbitrarily choosing 64 nodes for hidden layer (probably too small)
		
		#TODO- add gaussian noise to 3rd layer in network
			#Model has access to params to adjust this
		self.fc3 = nn.Linear(nodes2, action_size)  #create noisy layer and replace nn.Linear with it

		# self.m = nn.Sigmoid()
		self.m = nn.Tanh()

		#BatchNorm1D normalizes data to 0 mean and unit variance
		self.bn1 = nn.BatchNorm1d(nodes1)#, momentum = 0.1)
		self.bn2 = nn.BatchNorm1d(nodes2)#, momentum = 0.1)
		self.reset_parameters()

	def reset_parameters(self):
		#reset params - might be bad??
		self.fc1.weight.data.uniform_(-1.5e-3, 1.5e-3)
		self.fc2.weight.data.uniform_(-1.5e-3, 1.5e-3)
		self.fc3.weight.data.uniform_(-3e-3, 3e-3)

	def forward(self, state):
		#was this
		# x = F.relu((self.bn1(self.fc1(state)))) 
		# x = F.relu((self.bn2(self.fc2(x))))

		#now this
		x = F.relu((self.fc1(state))) #batchNorm kills neurons in ReLu when it sends data below 0
		x = F.relu((self.fc2(x)))
		#or
		# x = F.leaky_relu((self.bn1(self.fc1(state)))) 
		# x = F.leaky_relu((self.bn2(self.fc2(x))))
		#or
		# x = F.leaky_relu((self.fc1(state)))
		# x = F.leaky_relu((self.fc2(x)))

		x = self.fc3(x)

		#was this
		return(self.m(x))

		#THIS IS WORKING???? 
		# x = torch.clamp(x,min = 0.001, max = 1)
		# x = torch.clamp(x, max = 1)
		# return(x) #-def want a linear activation function

	def save_checkpoint(self):
		torch.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		self.load_state_dict(torch.load("checkpoint/actorBest"))


#Trying this- I think critic needs to have action added in 2nd layer(?)
# class Critic(nn.Module):
# 	"""Critic (Value) Model.""" 
# 	def __init__(self, state_size = 6, action_size = 9, nodes1=200, nodes2 = 100): #was 400,300
# 		super(Critic, self).__init__()

# 		self.checkpoint_file = "checkpoint/critic"

# 		self.fc1 = nn.Linear(state_size, nodes1)
# 		self.bn1 = nn.BatchNorm1d(nodes1)
# 		self.fc2 = nn.Linear(nodes1 + action_size, nodes2)		
# 		self.fc3 = nn.Linear(nodes2, 1)
# 		self.fc3.weight.data.uniform_(-WFINAL, WFINAL)
		
# 		self.ReLU = nn.ReLU()


# 	def forward(self, state, action):
# 		"""critic network that maps (state, action) pairs -> Q-values."""
		
# 		h1 = self.ReLU(self.fc1(state))
# 		h1_norm = self.bn1(h1)
# 		h2 = self.ReLU(self.fc2(torch.cat([h1_norm, action], dim=1)))
# 		Qval = self.fc3(h2)
# 		return Qval

# 	def save_checkpoint(self):
# 		torch.save(self.state_dict(), self.checkpoint_file)

# 	def load_checkpoint(self):
# 		self.load_state_dict(torch.load("checkpoint/critic"))

#Was this:
#simple 2 HL critic
class Critic(nn.Module):
	"""Critic (Value) Model.""" 
	def __init__(self, state_size = 6, action_size = 9, nodes1=16, nodes2 = 16): #was 200, 100
		super(Critic, self).__init__()

		self.checkpoint_file = "checkpoint/critic"

		self.fc1 = nn.Linear(state_size+action_size, nodes1)
		self.fc2 = nn.Linear(nodes1, nodes2)
		self.fc3 = nn.Linear(nodes2, 1)

	def forward(self, state, action):
		"""critic network that maps (state, action) pairs -> Q-values."""
		
		inputVals = torch.cat((state, action),1)

		# x = F.relu(self.bn1(self.fc1(inputVals)))
		x = F.relu(self.fc1(inputVals))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def save_checkpoint(self):
		torch.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		self.load_state_dict(torch.load("checkpoint/criticBest"))


#pre 12/7/2020
# class Critic(nn.Module):
# 	"""Critic (Value) Model.""" 
# 	def __init__(self, state_size = 6, action_size = 9, s1_units=10, s2_units = 5 , a1_units=5): #was 1000, 1000
# 		super(Critic, self).__init__()
# 		self.fc1 = nn.Linear(state_size, s1_units)
# 		self.fc2 = nn.Linear(s1_units, s2_units)
# 		# self.fc3 = nn.Linear(fc2_units, 1)
# 		self.action_value = nn.Linear(action_size,a1_units)
# 		self.q = nn.Linear(s2_units,1)

# 		# self.m = nn.Sigmoid()
# 		self.m = nn.Tanh()

# 		self.fc2.weight.data.uniform_(-1.5e-3, 1.5e-3)
# 		self.q.weight.data.uniform_(-3e-3, 3e-3)
# 		#init weights
# 		# f1 = 1 / (np.sqrt(self.fc1.weight.data.size()[0]))
# 		# self.fc1.weight.data.uniform_(-f1, f1)
# 		# self.fc1.bias.data.uniform_(-f1, f1)
# 		# f2 = 0.002
# 		# self.fc2.weight.data.uniform_(-f2, f2)
# 		# self.fc2.weight.data.uniform_(-f2, f2)
# 		# f3 = 0.003
# 		# self.q.weight.data.uniform_(-f3, f3)
# 		# self.q.weight.data.uniform_(-f3, f3)

# 		#was this
# 		# self.bn1 = nn.LayerNorm(s1_units)
# 		# self.bn2 = nn.LayerNorm(s2_units)
# 		#changed to
# 		self.bn1 = nn.BatchNorm1d(s1_units)

# 	def forward(self, state, action):
# 		"""critic network that maps (state, action) pairs -> Q-values."""
# 		#was this
# 		state_value = F.relu(self.bn1(self.fc1(state)))
# 		state_value = self.fc2(state_value)
# 		# action_value = F.relu(self.action_value(action))
# 		action_value = self.action_value(action)
# 		state_action_value = F.relu(torch.add(state_value, action_value))

# 		#now this
# 		# state_value = self.m(self.bn1(self.fc1(state)))
# 		# state_value = self.fc2(state_value)
# 		# action_value = self.action_value(action)
# 		# state_action_value = self.m(torch.add(state_value, action_value))

# 		return self.q(state_action_value)