from statePredictor import statePredictor
import numpy as np
from model import Actor, Critic
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
from agent import Agent

#main script
#generates policy to determine friction parameters of robot


#init CUDA
if torch.cuda.is_available():
	device = torch.device("cuda:0")
	torch.set_default_tensor_type('torch.cuda.FloatTensor') 
	print("Running on GPU")
else:
	device = torch.device("cpu")
	torch.set_default_tensor_type('torch.FloatTensor')
	print("Running on the CPU")


#TODO optimize dt
dt = 0.5
trials = 1000

#init state predictors ---------------------
#ground truth model
gt = statePredictor()
gt.dt = dt #length of time between initial and final states
gt.numPts = 2 #only care about start and next state, no need for anything else
#estimated friction model
ef = statePredictor()
ef.dt = dt
ef.numPts = 2

#init NN------------------------------------
agent = Agent(6,9)
rews = np.zeros(trials)
agent_loss = np.zeros(trials)
critic_loss = np.zeros(trials)

for trial in range(trials):
	print("Trial #: ", trial)

	#get random initial states: start with zero velocities for now
	# states = np.random.rand(3)
	# gt.x0[:3] = states 
	# ef.x0[:3] = states
	states = np.random.rand(6)
	gt.x0 = states 
	ef.x0 = states

	states = torch.from_numpy(states).float() #convert to float tensor
	#use actor to determine actions based on states
	agent.actor.eval()

	with torch.no_grad(): #TODO- verify if I really want this
		action = agent.actor.forward(states.view(-1,6))
	agent.actor.train() #unlocks actor

	#bring back to cpu for running on model - not sure if this is necessary
	# act = action.cpu().detach().numpy()

	#plug actions chosen by actor network into enviornment
	ef.numerical_constants[12:] = action #in this case actions are friction parameters
	efStates = ef.predict()[1]

	# print("efStates = ", efStates)

	#calculate ground truth solution
	gtStates = gt.predict()[1]
	# print("gt.numerical_constants[12:] = ", gt.numerical_constants[12:])
	# print("gtStates = ", gtStates)

	#get error from enviornment
	reward = -1* np.sum((gtStates[:3] - efStates[:3])**2) #only care about position for now

	#plug state-action pair into critic network to get error

	#get output from critic network

	#get difference between critic and enviornment

	#updates actor and critic network
	#TODO figure out dones
	done = 1 #all steps are independant of previous steps(?) 
	agent.step(states, action, reward, efStates, done)
		#step calls agent.learn()
			#learn() calls critic forward and update
			#TODO- is this sufficient???

	rews[trial] = reward #.cpu().numpy()
	print("reward = ", rews[trial])
	agent_loss[trial] = agent.aLossOut
	critic_loss[trial] = agent.cLossOut

	#DEBUG
	if trial % 10 == 0:
		print("ef.numerical_constants[12:] = ", ef.numerical_constants[12:])
		print("gt.numerical_constants[12:] = ", gt.numerical_constants[12:])

		print("ef states = ", efStates)
		print("gt states = ", gtStates)

np.save("rewards.txt", rews)


#save values for actor loss, critic loss, rewards after each trial
	#potentially plot these values as simulation runs??

#save policy -> generate lookup table???
			 # -> MAKE FUNCTION FROM POLICY??? -> add to EOM??