import numpy as np
from statePredictor import statePredictor
from agent import Agent
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

#init CUDA
if torch.cuda.is_available():
	device = torch.device("cuda:0")
	torch.set_default_tensor_type('torch.cuda.FloatTensor') 
	print("Running on GPU")
else:
	device = torch.device("cpu")
	torch.set_default_tensor_type('torch.FloatTensor')
	print("Running on the CPU")

trialLim = 10
tickLim = 200
thresh = 0.01 #0.1 #1
#make sure these params are the same as checkpoint policy
action_scale = action_scale = np.array([25,25,25])
gravity = True
friction = False
fidelity = 0.01

agent = Agent(9,3)
agent.load_models()

sp = statePredictor()
sp.dt = fidelity
sp.numPts = 2
if gravity == False:
	sp.numerical_constants[11] = 0
if friction == False:
	sp.numerical_constants[12:] = 0

path = np.zeros([1,3])
goal_path = np.zeros([1,3])

#start at random pos
states = torch.randn(6)

count = torch.zeros(1)

tick = 0
running = True
for trial in range(trialLim):
	print("trial ", trial)
	path_temp = np.zeros([1,3])
	goal_path_temp = np.zeros([1,3])

	# states[3:] = 0
	goal_pos_start = 0.5*torch.randn(3) #periodic
	# goal_pos_start = states[:3] + torch.randn(3)

	#start from last position of previous trial
	next_states = states 

	#start near goal pos at zero velocity
	# next_states[:3] = goal_pos_start + 0.1*torch.randn(3)
	# next_states[3:] = 0

	tick = 0
	done = 0
	while done != 1:
		goal_pos = 0.5*torch.sin(count/50) + goal_pos_start #periodic
		# goal_pos = goal_pos_start #constant
		states = next_states.float()
		states = states.to(device)

		agent.actor.eval()
		with torch.no_grad():
			action = agent.actor(torch.cat((states,goal_pos), dim=0).unsqueeze(0))
		agent.actor.train() #unlocks actor

		# print("states = ", states, " action = ", action)

		sp.numerical_specified = action.cpu().detach().numpy()[0]*action_scale
		sp.x0 = states

		next_states = sp.predict()[1]
		path_temp = np.append(path_temp,np.array([next_states[:3]]),axis = 0)
		goal_path_temp = np.append(goal_path_temp,np.array([goal_pos.cpu().numpy()]), axis=0)

		next_states = torch.as_tensor(next_states)

		# print(next_states)

		dist = torch.sum(abs((goal_pos.cpu()-states[:3].cpu())**2))

		# if dist < thresh and tick>20:
		# 	print("did that thing you like")
		# 	path = np.append(path, path_temp, axis=0)
		# 	goal_path = np.append(goal_path, goal_path_temp, axis=0)
		# 	done = 1

		# if torch.sum(abs(states[3:])) > 10:
		# 	done = 1
		# 	print("blew up")

		#timeout
		if tick == tickLim:
			path = np.append(path, path_temp, axis=0)
			goal_path = np.append(goal_path, goal_path_temp, axis=0)
			done = 1

		tick += 1
		count += 1

np.save("path",path)
np.save("goal_path",goal_path)