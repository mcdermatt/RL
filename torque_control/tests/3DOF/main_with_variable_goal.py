from statePredictor import statePredictor
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
from agent import Agent
# from kinematics import FK

# doing reward from forward kinematics allows arm to spin an extra revolution ->bad 

fidelity = 0.05 # seconds per step
trials = 20000
doneThresh = 0.01 #stop trial if theta gets within this distance
maxTrialLen = 100
gravity = False
friction = False
# action_scale = 3
# action_scale = np.array([2,6,4]) #when gravity = True
action_scale = np.array([0.1,0.1,0.1]) #can be much smaller when not fighting gravity
# goal_pos = torch.Tensor([1,0.5,1.2]) #was cart, setting it to angles now
save_progress = True

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
if gravity == False:
	sp.numerical_constants[11] = 0
if friction == False:
	sp.numerical_constants[12:] = 0

agent = Agent(9,3)

count = 0 #for actor and critic loss vectors
actor_loss = np.zeros(trials*maxTrialLen)
critic_loss = np.zeros(trials*maxTrialLen)

rewardArr = np.zeros(1)
startDist = 0
endDist = 0
tick = 0
for trial in range(trials):
	print("took ", tick, " ticks")
	print("trial ", trial, " -------------------------------------")
	#get initial states
	goal_pos = torch.randn(3)
	states = torch.randn(6) #simplify problem - start only in quadrant 1
	# states[3:] = torch.randn(3)
	# states[3:] = torch.zeros(3) #start at zero velocity
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
		
		sp.numerical_specified[:] = action.cpu().detach().numpy()[0]*action_scale
		sp.x0 = states

		# next_states = torch.as_tensor(sp.predict()[1])
		next_states = sp.predict()[1]
		next_states = torch.as_tensor(next_states)
		states = torch.as_tensor(states)

		#BAD IDEA ALERT
		#convert joint space positions to cart dist
		# cart = FK(states[0].cpu().numpy(),states[1].cpu().numpy(),states[2].cpu().numpy()) #TODO convert to numpy(?)
		# dist = torch.sum(abs((goal_pos-cart)**2))

		#set dist to be for each joint angle - can more heavily weigh base joints later if needed
		dist = torch.sum(abs((goal_pos.cpu()-states[:3].cpu())**2))
		reward = -dist
		reward = torch.as_tensor(reward)

		if tick == 0:
			startDist = dist
		if tick == maxTrialLen:
			# reward -= 10 #punishment for not finihsing
			endDist = dist
			done = 1
		# if dist < doneThresh and (abs(sp.x0[1]) < 0.1): #actual goal for 1dof -> go to this position and stop
		if dist < doneThresh: #simple goal -> get to goal pos
			done = 1
		done = torch.as_tensor(done)

		# agent.step(states.cpu().numpy(), action.cpu().numpy(), reward.cpu().numpy(), next_states.cpu().numpy(), done.cpu().numpy())
		agent.step(torch.cat((states,goal_pos), dim=0).cpu().numpy(), action.cpu().numpy(), reward.cpu().numpy(), torch.cat((next_states,goal_pos), dim=0).cpu().numpy(), done.cpu().numpy())

		# print("states = ",states)
		# print("next_states = ", next_states)

		tick += 1

		actor_loss[count] = agent.aLossOut #straight from critic
		critic_loss[count] = agent.cLossOut	
		count += 1	

	# rewardArr = np.append(rewardArr, startDist - endDist)
	rewardArr = np.append(rewardArr, endDist)
	print("goal = ", goal_pos)
	print("improvement = ", startDist - endDist)
	print("dist  = ",dist, " action = ", action.cpu().detach().numpy()[0])
	print("last states = ", states)

	if trial % 10 == 0:
		if save_progress:
			agent.save_models()
			np.savetxt("rewards", rewardArr)

			np.save("actor_loss",actor_loss)
			np.save("critic_loss",critic_loss)
