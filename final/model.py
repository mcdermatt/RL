import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Actor(nn.Module): #create actor class and inherit from nn.Module
	def __init__(self, state_size = 6, action_size = 9, nodes1 = 5000, nodes2 = 2500): #was 1000, 1000
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
		# x = F.relu(self.bn1(self.fc1(state))) 
		# x = F.relu(self.bn2(self.fc2(x)))
		x = self.m(self.fc1(state)) 
		x = self.m(self.fc2(x))

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
		# return(F.torch.tanh(x))
		
		#want sigmoid NOT tanh since fric will never be negative
		return(self.m(x)*10) #MULTIPLY BY 10 SINCE MAX VALUES OF FRIC CAN BE >> 1


class Critic(nn.Module):
	"""Critic (Value) Model.""" 
	def __init__(self, state_size = 6, action_size = 9, fc1_units=5000, fc2_units=2500): #was 1000, 1000
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