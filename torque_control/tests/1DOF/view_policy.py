import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from statePredictor import statePredictor
from time import sleep
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

trialLim = 250

#make sure these params are the same as checkpoint policy
action_scale = 3 #0.01 #3

agent = Agent(3,1) #pos, vel, goal pos
agent.load_models()

sp = statePredictor()
sp.dt = 0.01
sp.numPts = 2

#EASY MODE 
sp.numerical_constants[5:] = 0 #disable friction
# sp.numerical_constants[4] = 0 #no gravity

fig = plt.figure()
ax = fig.add_subplot(111, xlim=(-1,1), ylim=(-1,1), zlim=(-1,1), projection='3d', autoscale_on=False)
ax.grid(False)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
plt.xlabel("x",fontdict=None,labelpad=None)
plt.ylabel("y",fontdict=None,labelpad=None)
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')	
base, = plt.plot([0],[0],[0],'bo', markersize = 8)

tick = 0
running = True
while running:
	states = torch.randn(2)
	goal_pos = torch.randn(1)
	next_states = states

	tick = 0
	done = 0
	while done != 1:
		states = next_states.float()
		states = states.to(device)

		agent.actor.eval()
		with torch.no_grad():
			action = agent.actor(torch.cat((states,goal_pos), dim=0).unsqueeze(0))
		agent.actor.train() #unlocks actor

		sp.numerical_specified[0] = action.cpu().detach().numpy()[0]*action_scale
		sp.x0 = states

		next_states = sp.predict()[1]
		next_states = torch.as_tensor(next_states)

		#only plot every other to keep time realistic
		if tick%3 == 0:
			link, = plt.plot([0,np.sin(states.cpu().numpy()[0])],[0,0],[0,np.cos(states.cpu().numpy()[0])], 'b-', lw = 6)
			goal, = plt.plot([np.sin(goal_pos.cpu().numpy()[0])],[0],[np.cos(goal_pos.cpu().numpy()[0])],'ro', markersize = 5)

			plt.draw()
			plt.pause(0.01)
			# plt.pause(0.03)
			link.remove()
			goal.remove()

		#timeout
		if tick == trialLim:
			done = 1

		tick += 1