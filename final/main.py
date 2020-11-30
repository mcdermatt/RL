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

#TODO
#	SWITCH optim.Adam to Radam

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
dt = 0.5 #was 0.1
trials = 5000

#init state predictors ---------------------
#ground truth model
gt = statePredictor()
gt.dt = dt #length of time between initial and final states
gt.numPts = 2 #only care about start and next state, no need for anything else
gt.x0 = np.zeros(6)

#simple case- all friction params are zero EXCEPT ONE
gt.numerical_constants[12:] = 0
gt.numerical_constants[14] = 10 #lock out elbow

#estimated friction model
ef = statePredictor()
ef.dt = dt
ef.numPts = 2
ef.x0 = np.zeros(6)
# ef.x0[1] = 1
# ef.x0[2] = 1

#init NN------------------------------------
agent = Agent(6,9)

rews = np.zeros(trials)
actor_loss = np.zeros(trials)
critic_loss = np.zeros(trials)

for trial in range(trials):

	#get random initial states: start with zero velocities for now
	# states = np.random.rand(3)
	# gt.x0[:3] = states 
	# ef.x0[:3] = states

	#CONFIGURATION DEPENDANT - might actually help get static consts..
	states = torch.randn(6)
	r1 = np.random.rand()
	r2 = np.random.rand()
	r3 = np.random.rand()
	if r1 > 0.9:
		states[3] = 0 #set starting velocity of j0 to zero
	if r2 > 0.9:
		states[4] = 0 #set starting velocity of j1 to zero
	if r3 > 0.9:
		states[5] = 0 #set starting velocity of j2 to zero
	#CONFIG INDEPENDANT
	# states = torch.Tensor([0,0,0,0,0,0])
	#test to see if changing up one joint is robust enough
	# states[1] = np.random.randn()

	states = states.to(device)
	gt.x0 = states 
	ef.x0 = states
	# print(type(gt.x0))

	#CONFIG INDEPENDANT - can't have no input states for Network so using 1 joint angle
	# states = np.random.rand(1) + 1
	# gt.x0[1] = states
	# ef.x0[1] = states	

	# states = torch.from_numpy(states).float() #convert to float tensor
	#use actor to determine actions based on states
	agent.actor.eval()
	with torch.no_grad(): #TODO- verify if I really want this
		#IMPORTANT- do NOT call forward func, calling actor uses __call__ which automatically
		#	handles the forward method
		# action = agent.actor(states).cpu().data.numpy() + np.random.randn(9)*0.05*(np.e**(-trial/(trials/10))) #decrease noise over time + agent.noise.sample()
		# action = agent.actor(states).cpu().data.numpy() + np.random.randn(9)*0.1*(np.e**(-trial/(trials/10))) #having problems with .device()
		action = agent.actor(states)
		# print(action)
	agent.actor.train() #unlocks actor

	#time decaying noise
	# action = action	+ torch.randn(9)*(np.e**(-trial/(trials/100)))
	#random shift by some %
	a = action
	action = action*(1 + torch.randn(9)*0.01) #fixed
	# action = action*(1 + torch.randn(9)*(np.e**(-trial/(trials/100))))	#time decaying

	#bring back to cpu for running on model - not sure if this is necessary
	# act = action.cpu().detach().numpy()

	#plug actions chosen by actor network into enviornment
	ef.numerical_constants[12:] = action.cpu().detach().numpy() #in this case actions are friction parameters
	efStates = ef.predict()[1]

	# print("efStates = ", efStates)

	#calculate ground truth solution
	gtStates = gt.predict()[1]
	# print("gt.numerical_constants[12:] = ", gt.numerical_constants[12:])
	# print("gtStates = ", gtStates)

	#BUG? this penalizes rewards for going under 1- I bet this is why loss was not improving...
	# reward = -np.log(np.sum((gtStates - efStates)**2))
	# reward = -np.log((np.sum(gtStates - efStates))**2)
	reward = -(np.sum(abs(gtStates - efStates)))**2


	reward = torch.as_tensor(reward)
	#send everything back to.(device)---------
	# states = states.to(device)
	# action = action.to(device)
	# reward = reward.to(device)
	# efStates = efStates.to(device)
	# done = done.to(device)
	
	efStates = torch.as_tensor(efStates)	#make tensor	

	#updates actor and critic network
	#TODO figure out dones
	done = 1 #all steps are independant of previous steps(?) 
	done = torch.as_tensor(done)
	# print(type(states))
	# print(type(action))
	# print(type(reward))
	# print(type(efStates))
	# print(type(done))

	agent.step(states.cpu().numpy(), action.cpu().numpy(), reward.cpu().numpy(), efStates.cpu().numpy(), done.cpu().numpy())

		#step calls agent.learn()
			#learn() calls critic forward and update
			#TODO- is this sufficient???

	rews[trial] = reward #.cpu().numpy()
	actor_loss[trial] = agent.aLossOut #straight from critic
	critic_loss[trial] = agent.cLossOut #error between critic and enviorment

	#DEBUG
	if trial % 10 == 0:
		print("Trial #: ", trial, "----------------------------------------------")
		print("action = ", a)
		print("action + noise = ", ef.numerical_constants[12:])
		print("ground truth = ", gt.numerical_constants[12:])

		print("ef states = ", efStates)
		print("gt states = ", gtStates)

		print("actor loss =", agent.aLossOut)
		print("critic loss =", agent.cLossOut)

		print("reward = ", rews[trial])


	if trial % 50 == 0:
		np.save("rewards", rews)
		np.save("actor_loss", actor_loss)
		np.save("critic_loss", critic_loss)

		torch.save(agent.actor.state_dict(), 'checkpoint/checkpoint_actor.pth')
		torch.save(agent.critic.state_dict(), 'checkpoint/checkpoint_critic.pth')
		torch.save(agent.actor_target.state_dict(), 'checkpoint/checkpoint_actor_t.pth')
		torch.save(agent.critic_target.state_dict(), 'checkpoint/checkpoint_critic_t.pth')

np.save("rewards", rews)
np.save("actor_loss", actor_loss)
np.save("critic_loss", critic_loss)


#save values for actor loss, critic loss, rewards after each trial
	#potentially plot these values as simulation runs??

#save policy -> generate lookup table???
			 # -> MAKE FUNCTION FROM POLICY??? -> add to EOM??