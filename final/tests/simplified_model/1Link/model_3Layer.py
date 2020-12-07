import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Actor(nn.Module): #create actor class and inherit from nn.Module
	def __init__(self, state_size = 6, action_size = 9, nodes1 = 20, nodes2 = 10, nodes3 = 5):
		super(Actor,self).__init__() #need to run this because init func of nn.Module is not run upon inherit

		#Linear is a simple flat fuly connected
		self.fc1 = nn.Linear(state_size, nodes1) #input current 13 state observations
		self.fc2 = nn.Linear(nodes1, nodes2)  #arbitrarily choosing 64 nodes for hidden layer (probably too small)
		self.fc3 = nn.Linear(nodes2,nodes3)
		self.fc4 = nn.Linear(nodes3, action_size)

		self.m = nn.Sigmoid()
		# self.m = nn.Tanh()

		#BatchNorm1D normalizes data to 0 mean and unit variance
		self.bn1 = nn.BatchNorm1d(nodes1, momentum = 0.1)
		self.bn2 = nn.BatchNorm1d(nodes2, momentum = 0.1)
		self.bn3 = nn.BatchNorm1d(nodes3, momentum = 0.1)
		self.reset_parameters()

	def reset_parameters(self):
		#reset params - might be bad??
		self.fc1.weight.data.uniform_(-1.5e-3, 1.5e-3)
		self.fc2.weight.data.uniform_(-1.5e-3, 1.5e-3)
		self.fc3.weight.data.uniform_(-3e-3, 3e-3)

	def forward(self, state):
		#this		
		# x = F.relu((self.fc1(state))) #batchNorm kills neurons in ReLu when it sends data below 0
		# x = F.relu((self.fc2(x)))
		# x = F.relu((self.fc3(x)))
		#or
		x = self.m((self.bn1(self.fc1(state)))) #batchNorm kills neurons in ReLu when it sends data below 0
		x = self.m((self.bn2(self.fc2(x))))
		x = self.m((self.bn3(self.fc3(x))))

		x = self.fc4(x)

		# return(x) #-def want a linear activation function
		return(self.m(x))

class Critic(nn.Module):
	"""Critic (Value) Model.""" 
	def __init__(self, state_size = 6, action_size = 9, nodes1=20, nodes2 = 10 , nodes3 =5): #was 1000, 1000
		super(Critic, self).__init__()

		self.fc1 = nn.Linear(state_size+action_size, nodes1)
		self.fc2 = nn.Linear(nodes1, nodes2)
		self.fc3 = nn.Linear(nodes2, nodes3)
		self.fc4 = nn.Linear(nodes3, 1)

		#was this 
		# self.fcs1 = nn.Linear(state_size, s1_units)
		# self.fcs2 = nn.Linear(s1_units, s2_units)

		# self.fca1 = nn.Linear(action_size,a1_units)
		# self.fc1 = nn.Linear(s2_units,1)

		# # self.m = nn.Sigmoid()
		# self.m = nn.Tanh()

		# self.fcs2.weight.data.uniform_(-1.5e-3, 1.5e-3)
		# self.fc1.weight.data.uniform_(-3e-3, 3e-3)
		# # self.bn1 = nn.LayerNorm(s1_units)
		# # self.bn2 = nn.LayerNorm(s2_units)
		# #changed to
		# self.bn1 = nn.BatchNorm1d(s1_units)

	def forward(self, state, action):
		"""critic network that maps (state, action) pairs -> Q-values."""
		
		inputVals = torch.cat((state, action),1)
		# x = F.relu(self.bn1(self.fc1(inputVals)))
		x = F.relu(self.fc1(inputVals))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return x

		#was this
		# state_value = F.relu(self.bn1(self.fcs1(state)))
		# state_value = self.fcs2(state_value)
		# # action_value = F.relu(self.action_value(action))
		# action_value = self.fca1(action)
		# state_action_value = F.relu(torch.add(state_value, action_value))

		# x = F.relu(self.fc1(state_action_value))
		# x = self.fc2(x)

		# return x