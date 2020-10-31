from model import Actor, Critic
from ragdoll import ragdoll
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
from agent import Agent

#TODO 
#	Convert Net() to actor critic network - this will handle issues of calculating loss
#	allow script to run with viz = False
#	figure out loss function
#	standardize inputs and outputs

# if torch.cuda.is_available():
# 	device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
# 	torch.set_default_tensor_type('torch.cuda.FloatTensor') 
# 	print("Running on GPU")
# else:
# 	device = torch.device("cpu")
# 	torch.set_default_tensor_type('torch.FloatTensor')
# 	print("Running on the CPU")
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

device = torch.device("cpu")

#init agent (state size, action size)
agent = Agent(13,5)

trials = 1000 #repeat simulation Epoch times
learning_rate = 0.001
for trial in range(trials):
	#resets simulation
	body = ragdoll(viz = True, arms = False, playBackSpeed = 10)

	while body.fallen == False:
		#get states of ragdoll
		states = body.get_states()
		states = states.to(device) #send to GPU

		#get action from actor
		agent.actor_local.eval() #lock actor_local
		with torch.no_grad():
			action = agent.actor_local.forward(states.view(-1,13))
		agent.actor_local.train() #unlock actor_local

		body.activate_joints(action[0,0],action[0,1],action[0,2],action[0,3],action[0,4])
		body.tick()
		states_next = body.get_states()
		#
		agent.step(states,action,body.reward,states_next,body.fallen)




#x is the input (states of robot joints)
# input = Variable(torch.rand(13),requires_grad = False)
# y = net.forward(x.view(-1,18))

# print("input = ",input)
#first weight
# print("w1 = ", net.fc1.weight.data )

# trials = 10 #repeat simulation Epoch times
# learning_rate = 0.001
# for trial in range(trials):

# 	print("trial # ", trial)

# 	body = ragdoll(viz = True, arms = True, playBackSpeed = 1)
# 	# input = body.get_states()
# 	# input = input.to(device)
# 	# body.tick() #tick once to randomize starting states slightly

# 	# simulate one timestep 
# 	while body.fallen == False:
		
# 		#get joint states
# 		input = body.get_states()
# 		input = input.to(device)
# 		# print("input = ", input)

# 		# feed states into NN and get out torques as per current weights
# 		output = actor.forward(input.view(-1,13)) #need to shift to allow negative torque
		
# 		# print("max = ", torch.max(output, dim = 1).values[0])
# 		# output = output/ torch.max(output, dim = 1).values[0] # TEMPORARY normalize
		
# 		#saturate torques
# 		sig = nn.Sigmoid()
# 		output = sig(output) - 0.5
# 		# print("output = ", output)
# 		#feed torques into simulation and step forward one timestep
# 		body.activate_joints(output[0,0],output[0,1],output[0,2],output[0,3],output[0,4])
# 		#simulate and update visualization
# 		body.tick()

# 	#get reward heuristic
# 	reward = body.reward
# 	#zero grads
# 	actor.zero_grad()

# 	#calculate loss - account for energy expended(?)	
# 	# temp = torch.Tensor()
# 	# loss = F.nll_loss(output,temp)

# 	# print("loss = ", loss)

# 	#backward pass
# 	# loss.backward()

# 	#update weights	
# 	actor_optim.step()
	
# 	#update learning rate???



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
# #took ~2.59s on GPU for 10000 trials - ~vErY NiCe~