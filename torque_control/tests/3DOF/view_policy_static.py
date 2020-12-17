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
from kinematics import *

#init CUDA
if torch.cuda.is_available():
	device = torch.device("cuda:0")
	torch.set_default_tensor_type('torch.cuda.FloatTensor') 
	print("Running on GPU")
else:
	device = torch.device("cpu")
	torch.set_default_tensor_type('torch.FloatTensor')
	print("Running on the CPU")

trialLim = 100

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

fig = plt.figure()
ax = fig.add_subplot(111, xlim=(-2,2), ylim=(-2,2), zlim=(-2,2), projection='3d', autoscale_on=False)
ax.grid(False)
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_zticklabels([])
plt.xlabel("x",fontdict=None,labelpad=None)
plt.ylabel("y",fontdict=None,labelpad=None)
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')	
base, = plt.plot([0],[0],[0],'go', markersize = 8)

tick = 0
running = True
while running:
	states = torch.randn(6)
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
		
		xElbow = np.sin(next_states[1])*np.sin(next_states[0])
		zElbow = np.cos(next_states[1])
		yElbow = np.sin(next_states[1])*np.cos(next_states[0])
		
		xEE, zEE, yEE = FK(next_states[0],next_states[1],next_states[2])
		goal_pos_cart = FK(goal_pos.cpu().numpy()[0],goal_pos.cpu().numpy()[2],goal_pos.cpu().numpy()[1])

		# xEE = xElbow + np.sin(next_states[1]+next_states[2])*np.sin(next_states[0])
		# zEE = yElbow + np.cos(next_states[1]+next_states[2])
		# yEE = zElbow + np.sin(next_states[1]+next_states[2])*np.cos(next_states[0])


		
		next_states = torch.as_tensor(next_states)



		Xs = [0, xElbow, xEE]
		Ys = [0, yElbow, yEE]
		Zs = [0, zElbow, zEE]
		# Xs = [0, xEE]
		# Ys = [0, yEE]
		# Zs = [0, zEE]

		print(next_states)

		# link, = plt.plot(Xs,Ys,Zs, 'b-', lw = 6)
		EE, = plt.plot([xEE],[yEE],[zEE],'bo',markersize = 8) #temp
		goal, = plt.plot([goal_pos_cart[0]],[goal_pos_cart[2]],[goal_pos_cart[2]],'ro', markersize = 5)

		plt.draw()
		plt.pause(0.1)
		# plt.pause(0.03)
		# link.remove()
		EE.remove()
		goal.remove()

		#timeout
		if tick == trialLim:
			done = 1

		tick += 1