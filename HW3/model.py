import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#TODO
# Take in input size from ragdoll model so we can add arms later
# Deep Deterministic Policy Gradient - Need Actor Critic network
# Add Replay Buffer

class Actor(nn.Module): #create actor class and inherit from nn.Module
	def __init__(self, state_size = 13, action_size = 5, nodes = 500):
		super(Actor,self).__init__() #need to run this because init func of nn.Module is not run upon inherit

		#Linear is a simple flat fuly connected
		self.fc1 = nn.Linear(state_size, nodes) #input current 13 state observations
		self.fc2 = nn.Linear(nodes, nodes)  #arbitrarily choosing 64 nodes for hidden layer (probably too small)
		self.fc3 = nn.Linear(nodes, action_size)  #output 5 torque values

		self.m = nn.Sigmoid()

	def forward(self, state):
		# x = F.relu(self.fc1(state)) 
		# x = F.relu(self.fc2(x))
		x = self.m(self.fc1(state)) 
		x = self.m(self.fc2(x))


		x = self.fc3(x)

    	#do not have to make output a certain function
    	#	need to map values of output layer to [0,1] for each element
    	#look into nn.BatchNorm1d()

		return(self.m(x))
		# return F.log_softmax(x, dim = 1)

class Critic(nn.Module):
	def __init__(self, state_size = 13, action_size = 5, nodes = 500):
		super(Critic,self).__init__()

		#use only fully connected layers (not using any image data)
		self.fcs1 = nn.Linear(state_size,nodes) #input is state observations
		self.fcs2 = nn.Linear(nodes,nodes)
		self.fcs3 = nn.Linear(nodes,1) # only one value for output?

		self.fca1 = nn.Linear(action_size,nodes)

		self.fc1 = nn.Linear(nodes,1) #outputs single value -> Q

		#define bn1 func
		self.bn1 = nn.BatchNorm1d(nodes)

	def forward(self, state, action):
		"""(state, action) -> Q"""
		xs = F.relu(self.bn1(self.fcs1(state)))
		xs = F.relu(self.fcs2(xs))
		xa = self.fca1(action)
		x = F.relu(torch.add(xs,xa))

		return(self.fc1(x))