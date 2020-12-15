from statePredictor import statePredictor
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
from agent import Agent

fidelity = 0.1 #0.01 # seconds per step
trials = 100
doneThresh = 0.1 #stop trial of theta gets within this distance
maxTrialLen = 100
action_scale = 3

#init CUDA
if torch.cuda.is_available():
	device = torch.device("cuda:0")
	torch.set_default_tensor_type('torch.cuda.FloatTensor') 
	print("Running on GPU")
else:
	device = torch.device("cpu")
	torch.set_default_tensor_type('torch.FloatTensor')
	print("Running on the CPU")


sp = statePredictor()
sp.dt = fidelity
sp.numPts = 2

#init arrays for tracking results
# rewards = np.zeros(trials*numSteps) 
# actor_loss = np.zeros(trials*numSteps)
# critic_loss = np.zeros(trials*numSteps)
# action_vec = np.zeros(trials*numSteps)
tick = 0

agent = Agent(3,1) #pos, vel, goal_pos

for trial in range(trials):
	print("took ", tick, " ticks")
	print("trial ", trial, " -------------------------------------")
	#get initial states
	goal_pos = torch.randn(1)
	states = torch.randn(2)
	# states = torch.zeros(2)
	next_states = states

	tick = 0
	done = 0
	while done != 1:

		states = next_states.float()
		states = states.to(device)

		#decide on action given states
		agent.actor.eval()
		with torch.no_grad():
			action = agent.actor(torch.cat((states,goal_pos), dim=0).unsqueeze(0))
		agent.actor.train() #unlocks actor

		# print("states = ",states, " action = ", action.cpu().detach().numpy()[0], " goal = ", goal_pos)
		
		sp.numerical_specified[0] = action.cpu().detach().numpy()[0]*action_scale
		sp.x0 = states

		# next_states = torch.as_tensor(sp.predict()[1])
		next_states = sp.predict()[1]
		next_states = torch.as_tensor(next_states)
		states = torch.as_tensor(states)
		reward = -abs(states[0] - goal_pos)
		reward = torch.as_tensor(reward)

		if tick == maxTrialLen:
			reward -= 10
			done = 1
		if (abs(sp.x0[0] - goal_pos)) < doneThresh and (abs(sp.x0[1]) < 0.1): #actual goal for 1dof -> go to this position and stop
		# if abs(sp.x0[0]) > goal_pos and abs(sp.x0[1]) < 0.1 : #simple goal -> get 2.5 rad away from 0 and stop moving
			done = 1
		done = torch.as_tensor(done)

		agent.step(torch.cat((states,goal_pos), dim=0).cpu().numpy(), action.cpu().numpy(), reward.cpu().numpy(), torch.cat((next_states,goal_pos), dim=0).cpu().numpy(), done.cpu().numpy())
		
		# print("states = ",states)
		# print("next_states = ", next_states)

		tick += 1
	print("goal = ", goal_pos, " states = ",states, " action = ", action.cpu().detach().numpy()[0])
