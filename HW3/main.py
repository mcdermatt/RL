from net import Net
from ragdoll import ragdoll
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
input = Variable(torch.rand(13),requires_grad = False)
# y = net.forward(x.view(-1,18))

print("input = ",input)
#first weight
# print("w1 = ", net.fc1.weight.data )

Epochs = 10 #repeat simulation Epoch times
learning_rate = 0.001
for epoch in range(Epochs):

	print("Epoch # ", epoch)

	body = ragdoll(viz = True, arms = True, playBackSpeed = 1)
	body.tick() #tick once to randomize starting states slightly

	# simulate one timestep 
	while body.fallen == False:
		
		#get joint states
		input = body.get_states()
		input = input.to(device)
		# print("input = ", input)

		# feed states into NN and get out torques as per current weights
		output = net.forward(input.view(-1,13)) #need to shift to allow negative torque
		
		# print("max = ", torch.max(output, dim = 1).values[0])
		# output = output/ torch.max(output, dim = 1).values[0] # TEMPORARY normalize
		
		#saturate torques
		sig = nn.Sigmoid()
		output = sig(output) - 0.5
		# print("output = ", output)
		#feed torques into simulation and step forward one timestep
		body.activate_joints(output[0,0],output[0,1],output[0,2],output[0,3],output[0,4])
		#simulate and update visualization
		body.tick()

	#get reward heuristic

	#calculate loss - account for energy expended(?)

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