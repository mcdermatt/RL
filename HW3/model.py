import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

#TODO
# Take in input size from ragdoll model so we can add arms later
# Deep Deterministic Policy Gradient - Need Actor Critic network
# Add Replay Buffer

class Actor(nn.Module): #create actor class and inherit from nn.Module
	def __init__(self, state_size = 13, action_size = 5, nodes1 = 1600, nodes2 = 800): #nodes1 = 2048, nodes2 = 1024):
		super(Actor,self).__init__() #need to run this because init func of nn.Module is not run upon inherit

		#Linear is a simple flat fuly connected
		self.fc1 = nn.Linear(state_size, nodes1) #input current 13 state observations
		self.fc2 = nn.Linear(nodes1, nodes2)  #arbitrarily choosing 64 nodes for hidden layer (probably too small)
		self.fc3 = nn.Linear(nodes2, action_size)  

		self.m = nn.Sigmoid()

		self.bn1 = nn.BatchNorm1d(nodes1)
		self.bn2 = nn.BatchNorm1d(nodes2)

		#reset params - might be bad??
		# self.fc1.weight.data.uniform_(-1.5e-3, 1.5e-3)
		# self.fc2.weight.data.uniform_(-1.5e-3, 1.5e-3)
		# self.fc3.weight.data.uniform_(-3e-3, 3e-3)

	def forward(self, state):
		x = F.relu(self.bn1(self.fc1(state))) 
		x = F.relu(self.bn2(self.fc2(x)))
		# x = self.m(self.fc1(state)) 
		# x = self.m(self.fc2(x))

		# x = F.relu(self.bn2(self.fc3(x)))
		# x = self.fc4(x)

		x = self.fc3(x)
		# print("x = ", x)
    	#do not have to make output a certain function
    	#	need to map values of output layer to [0,1] for each element
    	#look into nn.BatchNorm1d()

		
		# print("action = ", self.m(x))
		# return(self.m(x))
		# print("action = ", F.torch.tanh(x)) # tanh -> [-1, 1]
		return(F.torch.tanh(x))

class Critic(nn.Module):
	"""Critic (Value) Model.""" 
	def __init__(self, state_size, action_size, fc1_units=1600, fc2_units=800):
		super(Critic, self).__init__()
		self.fc1 = nn.Linear(state_size, fc1_units)
		self.fc2 = nn.Linear(fc1_units, fc2_units)
		# self.fc3 = nn.Linear(fc2_units, 1)
		self.action_value = nn.Linear(action_size,fc2_units)
		self.q = nn.Linear(fc2_units,1)

		f1 = 1 / (np.sqrt(self.fc1.weight.data.size()[0]))
		self.fc1.weight.data.uniform_(-f1, f1)
		self.fc1.bias.data.uniform_(-f1, f1)
		f2 = 0.002
		self.fc2.weight.data.uniform_(-f2, f2)
		self.fc2.weight.data.uniform_(-f2, f2)
		f3 = 0.003
		self.q.weight.data.uniform_(-f3, f3)
		self.q.weight.data.uniform_(-f3, f3)

		self.bn1 = nn.LayerNorm(fc1_units)
		self.bn2 = nn.LayerNorm(fc2_units)

	def forward(self, state, action):
		"""critic network that maps (state, action) pairs -> Q-values."""
		state_value = F.relu(self.bn1(self.fc1(state)))
		state_value = self.fc2(state_value)
		state_value = self.bn2(state_value)

		action_value = F.relu(self.action_value(action))
		state_action_value = F.relu(torch.add(state_value, action_value))
		state_action_value = self.q(state_action_value)

		# x = torch.cat((xs, action), dim=1)


		# x = F.relu(self.fc2(x))


		# return self.fc3(x)
		return state_action_value


#from towards data science article
# class Critic(nn.Module):
#     def __init__(self, input_size = 18, hidden_size=512, output_size=1):
#         super(Critic, self).__init__()
#         self.linear1 = nn.Linear(input_size, hidden_size)
#         self.linear2 = nn.Linear(hidden_size, hidden_size)
#         self.linear3 = nn.Linear(hidden_size, output_size)

#     def forward(self, state, action):
#         """
#         Params state and actions are torch tensors
#         """
#         x = torch.cat([state, action], 1)
#         x = F.relu(self.linear1(x))
#         x = F.relu(self.linear2(x))
#         x = self.linear3(x)

#         return x

#Shiva Verma's strategy: seperated action and state values
# class Critic(nn.Module):
# 	def __init__(self, state_size = 13, action_size = 5, nodes1 = 1024, nodes2 = 512):
# 		super(Critic,self).__init__()

# 		#use only fully connected layers (not using any image data)
# 		self.fcs1 = nn.Linear(state_size,nodes1) #input is state observations
# 		self.fcs2 = nn.Linear(nodes1,nodes2)
# 		# self.fcs3 = nn.Linear(nodes,1) # only one value for output?

# 		self.fca1 = nn.Linear(action_size,nodes2)
# 		self.fc1 = nn.Linear(nodes2,1) #outputs single value -> Q

# 		#define bn1 func
# 		self.bn1 = nn.BatchNorm1d(nodes1)

# 		#reset params
# 		self.fcs2.weight.data.uniform_(-1.5e-3,1.5e3)
# 		self.fc1.weight.data.uniform_(-3e-3,3e-3)

# 	def forward(self, state, action):
# 		"""(state, action) -> Q"""
# 		xs = F.relu(self.bn1(self.fcs1(state)))
# 		xs = self.fcs2(xs)
# 		xa = self.fca1(action)
# 		x = F.relu(torch.add(xs,xa))

# 		return(self.fc1(x))