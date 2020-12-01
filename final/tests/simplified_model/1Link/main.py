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
#	+180deg vs -180deg

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
dt = .5 # was 0.25
trials = 5000

#init state predictors ---------------------
#ground truth model
gt = statePredictor()
gt.dt = dt #length of time between initial and final states
gt.numPts = 100 #only care about start and next state, no need for anything else

#simple case- all friction params are zero EXCEPT ONE
# gt.numerical_constants[9:] = 0
# gt.numerical_constants[14] = 10 #max out damping

#estimated friction model
ef = statePredictor()
ef.dt = dt
ef.numPts = 100
# ef.x0[1] = 1
# ef.x0[2] = 1

#init NN------------------------------------
agent = Agent(2,3) #1DOF

rews = np.zeros(trials)
actor_loss = np.zeros(trials)
critic_loss = np.zeros(trials)

for trial in range(trials):

	#get random initial states: start with zero velocities for now
	# states = np.random.rand(3)
	# gt.x0[:3] = states 
	# ef.x0[:3] = states

	#CONFIGURATION DEPENDANT - might actually help get static consts..
	states = torch.randn(2)
	r1 = np.random.rand()
	if r1 > 0.9:
		states[1] = 0 #set starting velocity to zero
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
	
	a = action #save true action for printing later

	#add noise - this is how it is off policy
	action = action	+ torch.randn(3)*(0.25)*(np.e**(-trial/(trials/5)))
	# action = action*(1 + torch.randn(3)*0.01) #fixed
	# action = action*(1 + torch.randn(9)*(np.e**(-trial/(trials/100))))	#time decaying

	#bring back to cpu for running on model - not sure if this is necessary
	# act = action.cpu().detach().numpy()

	#zero out negative actions
	action[action < 0] = 0

	#plug actions chosen by actor network into enviornment
	ef.numerical_constants[5:] = action.cpu().detach().numpy() #in this case actions are friction parameters

	efStatesVec = ef.predict()
	efStates = efStatesVec[-1]
	gtStatesVec = gt.predict()
	gtStates = gtStatesVec[-1]

	# print("gt.numerical_constants[12:] = ", gt.numerical_constants[12:])
	# print("gtStates = ", gtStates)

	reward = -(np.sum(abs(gtStatesVec[:,0] - efStatesVec[:,0])))
	# reward = -(np.sum((abs(gtStatesVec[:,0] - efStatesVec[:,0]))**2)) #makes critic loss buggy

	reward = torch.as_tensor(reward)
	
	efStates = torch.as_tensor(efStates)	#make tensor	

	#updates actor and critic network
	done = 1 #all steps are independant of previous steps(?) 
	done = torch.as_tensor(done)

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
		print("action + noise = ", ef.numerical_constants[5:])
		print("ground truth = ", gt.numerical_constants[5:])

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