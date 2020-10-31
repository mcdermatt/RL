from model import Actor
from ragdoll import ragdoll
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time

#TODO 
#	Convert Net() to actor critic network - this will handle issues of calculating loss
#	allow script to run with viz = False
#	figure out loss function
#	standardize inputs and outputs
#	stop human from getting stuck in a split

if torch.cuda.is_available():
	device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
	torch.set_default_tensor_type('torch.cuda.FloatTensor') 
	print("Running on GPU")
else:
	device = torch.device("cpu")
	torch.set_default_tensor_type('torch.FloatTensor')
	print("Running on the CPU")
torch.set_default_tensor_type('torch.cuda.FloatTensor')

net = Actor()
net = net.to(device)

optimizer = optim.Adam(net.parameters(), lr = 0.001) #net.parmeters controls what stuff in net() is adjusted (default is everything)

#x is the input (states of robot joints)
input = Variable(torch.rand(13),requires_grad = False)
# y = net.forward(x.view(-1,18))

# print("input = ",input)
#first weight
# print("w1 = ", net.fc1.weight.data )

trials = 10 #repeat simulation Epoch times
learning_rate = 0.001
for trial in range(trials):

	print("trial # ", trial)

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
	reward = body.reward
	#zero grads
	net.zero_grad()

	#calculate loss - account for energy expended(?)	
	# temp = torch.Tensor()
	# loss = F.nll_loss(output,temp)

	# print("loss = ", loss)

	#backward pass
	# loss.backward()

	#update weights	
	optimizer.step()
	
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