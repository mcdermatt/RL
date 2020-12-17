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
thresh = 0.5
#make sure these params are the same as checkpoint policy
action_scale = 0.1
goal_pos = torch.tensor([1,0.5,1.2])
gravity = False
friction = False

agent = Agent(6,3)
agent.load_models()

sp = statePredictor()
sp.dt = 0.1
sp.numPts = 2
if gravity == False:
	sp.numerical_constants[11] = 0
if friction == False:
	sp.numerical_constants[12:] = 0

path = np.zeros([1,3])
tick = 0
running = True
for trial in range(trialLim):
	print("trial ", trial)
	states = torch.randn(6)
	states[0] = states[0]*2
	states[2] = states[2]*2
	states = torch.abs(states) #pos only for now
	states[3:] = torch.zeros(3)
	next_states = states

	tick = 0
	done = 0
	while done != 1:
		states = next_states.float()
		states = states.to(device)

		agent.actor.eval()
		with torch.no_grad():
			action = agent.actor(states.unsqueeze(0))
		agent.actor.train() #unlocks actor

		# print("states = ", states, " action = ", action)

		sp.numerical_specified = action.cpu().detach().numpy()[0]*action_scale
		sp.x0 = states

		next_states = sp.predict()[1]
		path = np.append(path,np.array([next_states[:3]]),axis = 0)

		next_states = torch.as_tensor(next_states)

		# print(next_states)

		dist = torch.sum(abs((goal_pos.cpu()-states[:3].cpu())**2))
		if dist < thresh and torch.sum(abs(states[3:])) < 0.1:
			print("did that thing you like")
			done = 1

		if torch.sum(abs(states[3:])) > 10:
			done = 1
			print("blew up")

		#timeout
		if tick == tickLim:
			done = 1

		tick += 1

np.save("path",path)