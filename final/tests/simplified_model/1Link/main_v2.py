from statePredictor_v2 import statePredictor
import numpy as np
from model import Actor, Critic
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
from agent_v2 import Agent

#Second attempt at AC implementation on one link 1DOF system
#	use typical RL Reward function: expected sum of discount rewards for each state

#	This is important because it ties states together temporally punishes early state extra if later states perform poorly	

#TODO
#	add noise to critic- don't let it get too confident
#	add parameter space noise
#	decide between tanh and ReLu- 
#		Tanh bad: vanishing gradients
# 		ReLu bad: dead neurons if < 0 values fed in
#	Map output of RL Agent to realistic range of friction values
#		none of these friction constants are actually going to be as large as 1 or as small as 0
#	Record estimates of each value vs ground truth and plot over time
#	Add momentum??

#CRITIC LOSS IS STAGNATING- need to make sure critic updates once goal is passed and stuff starts getting worse again

#init CUDA
if torch.cuda.is_available():
	device = torch.device("cuda:0")
	torch.set_default_tensor_type('torch.cuda.FloatTensor') 
	print("Running on GPU")
else:
	device = torch.device("cpu")
	torch.set_default_tensor_type('torch.FloatTensor')
	print("Running on the CPU")

dt = 0.5 # was 0.5
trials = 25000
numSteps = 20

#init state predictors ---------------------
#ground truth model
gt = statePredictor()
#Generate all at once- This only works when friction is independant of state
gt.dt = dt #length of time between initial and final states
gt.numPts = numSteps+1 #generate full vector at beginning of each trial since params don't change
# gt.numerical_constants[-1] = 0.5 #add high damping

#estimated friction model
ef = statePredictor()
ef.dt = dt/numSteps #lol
ef.numPts = 2 #only generate 2 at a time since friction params are updated each trial

#init NN------------------------------------
agent = Agent(2,3) #1DOF
# agent = Agent(2,1)

rews = np.zeros(trials*numSteps)
actor_loss = np.zeros(trials*numSteps)
critic_loss = np.zeros(trials*numSteps)

for trial in range(trials):
	#init starting states
	states = torch.randn(2)
	r1 = np.random.rand()
	if r1 > 0.9:
		states[1] = 0 #set starting velocity to zero since random variable will never do so
	states = states.to(device)
	# states = states.unsqueeze(0).to(device) #need to unsqueeze to work with batchnorm1d
	gt.x0 = states 
	ef.x0 = states
	efStates = states #temp

	#get ground truth state vec all at once(?)
	gtStatesVec = gt.predict()

	for step in range(numSteps-1):
		#use actor to determine friction values based on states
		agent.actor.eval()
		with torch.no_grad():
			action = agent.actor(states.unsqueeze(0)) + 0.1*torch.randn(3, device = device)*(np.e**(-trial/(trials/50)))
			action = torch.clamp(action, 0,1)
		agent.actor.train() #unlocks actor
		
		a = action #save true action for printing later	
		# action = action	+ torch.randn(3)*(0.25)*(np.e**(-trial/(trials/10))) #add noise
		# action = action	+ torch.randn(1)*(0.25)*(np.e**(-trial/(trials/10))) #add noise
		action[action < 0] = 0 	#zero out negative actions

		#plug actions chosen by actor network into enviornment
		ef.numerical_constants[5:] = action.cpu().detach().numpy() #in this case actions are friction parameters
		# ef.numerical_constants[-1] = action.cpu().detach().numpy() #in this case actions are friction parameters
		ef.x0 = efStates
		efStates = ef.predict()[1]
		gtStates = gtStatesVec[step]

		# reward = -(np.sum(abs(gtStates - efStates)))
		# reward = -abs(gtStates[0]-efStates[0]) #pos
		reward = -abs(gtStates[1]-efStates[1]) #vel
		reward = torch.as_tensor(reward)
		efStates = torch.as_tensor(efStates)	

		if step == numSteps:
			done = 1 
		else:
			done = 0
		done = torch.as_tensor(done)

		# if step*(trial+1) % 10 == 0:
		# 	is_learning = True
		# else:
		# 	is_learning = False
		agent.step(states.cpu().numpy(), action.cpu().numpy(), reward.cpu().numpy(), efStates.cpu().numpy(), done.cpu().numpy())

		rews[trial*numSteps + step] = reward #.cpu().numpy()
		actor_loss[trial*numSteps + step] = agent.aLossOut #straight from critic
		critic_loss[trial*numSteps + step] = agent.cLossOut #error between critic and enviorment

	if trial % 10 == 0:
		print("Trial #: ", trial, "----------------------------------------------")
		print("action + noise = ", ef.numerical_constants[5:])
		print("ground truth = ", gt.numerical_constants[5:])

		print("ef states = ", efStates)
		print("gt states = ", gtStates)
		print("reward = ", reward)


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