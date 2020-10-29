from net import Net
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time

#TODO 
#	figure out loss function
#	integrate stepping with ragdoll class
#	standardize inputs and outputs

if torch.cuda.is_available():
	device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
	torch.set_default_tensor_type('torch.cuda.FloatTensor') 
	print("Running on GPU")
else:
	device = torch.device("cpu")
	torch.set_default_tensor_type('torch.FloatTensor')
	print("Running on the CPU")
torch.set_default_tensor_type('torch.cuda.FloatTensor')

net = Net()
net = net.to(device)

#x is the input (states of robot joints)
x = Variable(torch.rand(18),requires_grad = False)
net.forward(x.view(-1,18))

print("x = ",x)
#first weight
# print("w1 = ", net.fc1.weight.data )

Epochs = 10 #repeat simulation Epoch times
learning_rate = 0.001
for epoch in range(Epochs):
	# simulate one timestep 
	while body.fallen == False:
		
		# feed states into NN and get out torques as per current weights

		#feed torques into simulation and step forward one timestep

		#update visualization
	
	#get reward heuristic

	#calculate loss

	#zero grads

	#backward pass

	#update weights	

	
	#update learning rate???



# #testing how long it takes to run through net to get output given input
# start = time.time()
# for i in range(10):
# 	x = torch.rand(18)
# 	x = x.to(device)
# 	net.zero_grad()

# 	y = net.forward(x.view(-1,18))
# 	# print(y)
# 	# print(y.grad)

# stop = time.time()
# print("it took ", stop-start," seconds to run")
# #took ~2.59s on GPU for 10000 trials