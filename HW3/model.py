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
	def __init__(self, state_size = 13, action_size = 5):
		super(Actor,self).__init__() #need to run this because init func of nn.Module is not run upon inherit

		#Linear is a simple flat fuly connected
		self.fc1 = nn.Linear(state_size, 64) #input current 13 state observations
		self.fc2 = nn.Linear(64, 64)  #arbitrarily choosing 64 nodes for hidden layer (probably too small)
		self.fc3 = nn.Linear(64, action_size)  #output 5 torque values

	def forward(self, state):
    	#F.relu is rectified linear activation func
    	#   activation func is sigmoid- keeps output from exploding
    	#   attempt to model whether neuron is or is not firing
		x = F.relu(self.fc1(state)) 
		x = F.relu(self.fc2(x))
    	#for output we only want one neuron to be fully fired
		x = self.fc3(x)
		print(x)
    	#do not have to make output a certain function
    	#	need to map values of output layer to [0,1] for each element

    	#look into nn.BatchNorm1d()

		return(x)
		# return F.log_softmax(x, dim = 1)

class Critic(nn.Module):
	def __init__(self, state_size = 13, action_size = 5):
		super(Critic,self).__init__()

		#use only fully connected layers (not using any image data)
		self.fcs1 = nn.Linear(state_size,64) #input is state observations
		self.fcs2 = nn.Linear(64,64)
		self.fcs3 = nn.Linear(64,1) # only one value for output?

		self.fca1 = nn.Linear(action_size,64)

		self.fc1 = nn.Linear(64,1) #outputs single value -> Q

		#define bn1 func
		self.bn1 = nn.BatchNorm1d(64)

	def forward(self, state, action):
		"""(state, action) -> Q"""
		xs = F.relu(self.bn1(self.fcs1(state)))
		xs = F.relu(self.fcs2(xs))
		xa = self.fca1(action)
		x = F.relu(torch.add(xs,xa))

		return(self.fc1(x))