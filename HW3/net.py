import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#TODO
# Figure out how to deal with multiple outputs
# Figure out how to deal with continuous outputs
# Take in input size from ragdoll model so we can add arms later

class Net(nn.Module): #create net and inherit from nn.Module
	def __init__(self):
		super().__init__() #need to run this because init func of nn.Module is not run upon inherit

		#Linear is a simple flat fuly connected
		self.fc1 = nn.Linear(13, 64) #input current 18 states
		self.fc2 = nn.Linear(64, 64)  #arbitrarily choosing 64 nodes for hidden layer
		self.fc3 = nn.Linear(64, 5)  #output 5 torque values

	def forward(self, x):
    	#F.relu is rectified linear activation func
    	#   activation func is sigmoid- keeps output from exploding
    	#   attempt to model whether neuron is or is not firing
		x = F.relu(self.fc1(x)) 
		x = F.relu(self.fc2(x))
    	#for output we only want one neuron to be fully fired
		x = self.fc3(x)
    
    	#do not have to make output a certain function
    	#	need to map values of output layer to [0,1] for each element

    	#look into nn.BatchNorm1d()

		return(x)
		# return F.log_softmax(x, dim = 1)