import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#TODO
# Figure out optimal number of layers
# Figure out how to deal with multiple outputs
# Decide on inputs
# actor-critic network??

class Net(nn.Module): #create net and inherit from nn.Module
	def __init__(self):
		super.__init__(): #need to run this because init func of nn.Module is not run upon inherit

		#Linear is a simple flat fuly connected
        self.fc1 = nn.Linear(, 64) #when images are flattened they are 28*28 = 784 long
        self.fc2 = nn.Linear(64, 64)  #arbitrarily choosing 64 nodes for hidden layers
        self.fc3 = nn.Linear(64, 64) 
        self.fc4 = nn.Linear(64, 3**5)  #output layer is size 10 for digits 0-9

        def forward(self, x):
        #F.relu is rectified linear activation func
        #   activation func is sigmoid- keeps output from exploding
        #   attempt to model whether neuron is or is not firing
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #for output we only want one neuron to be fully fired
        x = self.fc4(x)
        
        return F.log_softmax(x, dim = 1)