import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Actor(nn.Module): #create actor class and inherit from nn.Module
	def __init__(self, state_size = 6, action_size = 9, nodes1 = 50):
		super(Actor,self).__init__() #need to run this because init func of nn.Module is not run upon inherit

		#Linear is a simple flat fuly connected
		self.fc1 = nn.Linear(state_size, nodes1) #input current 13 state observations
		self.fc2 = nn.Linear(nodes1, action_size)  #arbitrarily choosing 64 nodes for hidden layer (probably too small)
		
		# self.m = nn.Sigmoid()

		#BatchNorm1D normalizes data to 0 mean and unit variance
		# self.bn1 = nn.BatchNorm1d(nodes1, momentum = 0.1)
		# self.reset_parameters()

	# def reset_parameters(self):
		#reset params - might be bad??
		# self.fc1.weight.data.uniform_(0, 1.5e-3)
		# self.fc2.weight.data.uniform_(0, 1.5e-3)

	def forward(self, state):
		x = F.relu(self.fc1(state)) #batchNorm kills neurons in ReLu when it sends data below 0
		x = self.fc2(x)
		# return(self.m(x))
		return(x)

class Critic(nn.Module):
	"""Critic (Value) Model.""" 
	def __init__(self, state_size = 6, action_size = 9, nodes1 = 50):
		super(Critic, self).__init__()
		self.fc1 = nn.Linear(state_size + action_size, nodes1)
		self.fc2 = nn.Linear(nodes1, 1)
		# self.m = nn.Tanh()
		# self.bn1 = nn.BatchNorm1d(nodes1)

	def forward(self, state, action):
		"""critic network that maps (state, action) pairs -> Q-values."""

		inputVals = torch.cat((state, action),1)
		# x = F.relu(self.bn1(self.fc1(inputVals)))
		x = F.relu(self.fc1(inputVals))
		x = self.fc2(x)
		return x