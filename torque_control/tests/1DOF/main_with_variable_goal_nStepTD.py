from statePredictor import statePredictor
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
# from agent import Agent
from agent_nStepTD import Agent
import collections

#TODO - do I need to chop off the end??
#TODO learn how to use scheduler
#TODO dynamically lower maxTrialLen
#ONLY STEPPING ONCE AFTER EVERY TRIAL .... -> make so I step after every  n?
#TODO Figure out what to do about rewards -> use advantage?

fidelity = 0.01 #0.1 # seconds per step
trials = 1000 #50000
doneThresh = 0.01 #0.1 #stop trial if theta gets within this distance with low velocity
maxTrialLen = 500
action_scale = 3 
save_progress = True
n = 4 #100 #15 #30 #number of TD steps to take
discount_factor = 0.99 #0.5  #0.95

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

#EASY MODE 
sp.numerical_constants[5:] = 0 #disable friction
# sp.numerical_constants[4] = 0 #no gravity

#init arrays for tracking results
count = 0
rewardArr = np.zeros(1)

actor_loss = np.zeros(trials*maxTrialLen)
critic_loss = np.zeros(trials*maxTrialLen)
action_vec = np.zeros(trials*maxTrialLen)
tick = 0

agent = Agent(3,1) #pos, vel, goal_pos

for trial in range(trials):
	print("took ", tick, " ticks")
	print("trial ", trial, " -------------------------------------")
	#get initial states
	goal_pos = torch.randn(1)
	# goal_pos = torch.zeros(1) #easy mode
	states = torch.randn(2)
	next_states = states

	SARS_of_current_trial = collections.deque()

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
		reward = -(abs(states[0] - goal_pos)**2) #- 0.1*abs(states[1]) #test velocity
		reward = torch.as_tensor(reward)

		if tick == maxTrialLen: #timeout
			# reward -= 10 #punishment for not finihsing
			done = 1
		#reaches goal
		# if (abs(sp.x0[0] - goal_pos)) < doneThresh and (abs(sp.x0[1]) < 0.1): #actual goal for 1dof -> go to this position and stop
		# if abs(sp.x0[0]) > goal_pos and abs(sp.x0[1]) < 0.1 : #simple goal -> get 2.5 rad away from 0 and stop moving
			# done = 1
		
		done = torch.as_tensor(done)

		e = agent.memory.experience(torch.cat((states,goal_pos), dim=0).cpu().numpy(), action.cpu().numpy(), reward.cpu().numpy(), torch.cat((next_states,goal_pos), dim=0).cpu().numpy(), done.cpu().numpy())
		SARS_of_current_trial.append(e)

		#update rewards of past n trials
		# TODO while tick < n we need to only partially update
		# 		or if we're lazy just don't share those with the main replayBuffer...
		if tick > n:
			for i in range(1,n):

				agent.actor.eval()
				with torch.no_grad():
					next_action = agent.actor(torch.cat((states,goal_pos),dim=0).unsqueeze(0)) #unsqueeze: tensor([]) -> tensor([[]])
					# print("next action will be ", next_action)
					# next_action = next_action.cpu().numpy()
				agent.actor.train()

				agent.critic.eval()
				with torch.no_grad():
					next_states_and_goal = torch.cat((next_states,goal_pos),dim=0).unsqueeze(0).float()
					crit1 = agent.critic(next_states_and_goal,next_action).cpu().numpy()
					
					current_states_and_goal = torch.cat((states,goal_pos),dim=0).unsqueeze(0).float()
					crit2 = agent.critic(current_states_and_goal,action).cpu().numpy()
					# print("critic says ", crit2 )
				agent.critic.train()

				#advantage is not the right term but YOLO
				advantage = SARS_of_current_trial[tick-i].reward + SARS_of_current_trial[tick].reward*discount_factor**(i) \
																 + (discount_factor**i)*crit1 \
																 - crit2

				#want to do this:
				# SARS_of_current_trial[tick-i].reward += SARS_of_current_trial[tick].reward*discount_factor**(i)  
				#namedTuples (exp) are immutable so we need a workaround
				SARS_of_current_trial[tick-i] = agent.memory.experience(SARS_of_current_trial[tick-i].state, 
																		SARS_of_current_trial[tick-i].action, 
																		# SARS_of_current_trial[tick-i].reward + SARS_of_current_trial[tick].reward*discount_factor**(i), #was this
																		advantage,
																		SARS_of_current_trial[tick-i].next_state, 
																		SARS_of_current_trial[tick-i].done)

		tick += 1
		actor_loss[count] = agent.aLossOut #straight from critic
		critic_loss[count] = agent.cLossOut	
		count += 1	

	#get rid of first n elements in SARS_of_current_trial (lazy mode)
	if tick < n:
		min_of_n_and_t = tick
	else:
		min_of_n_and_t = n
	for _ in range(min_of_n_and_t):
		SARS_of_current_trial.popleft()

	#TODO figure out a way to do this that does not remove good results - right now I am just making the finish threshold really small
	for _ in range(n):
		try:
			SARS_of_current_trial.pop()
		except:
			pass

	#only occurs once a trial -> SLOW
	agent.step(SARS_of_current_trial)
		
	print("goal = ", goal_pos, " states = ",states, " action = ", action.cpu().detach().numpy()[0])
	
	rewardArr = np.append(rewardArr, reward.cpu().numpy())

	if trial % 10 == 0:
		if save_progress:
			agent.save_models()
			np.save("actor_loss",actor_loss)
			np.save("critic_loss",critic_loss)
			np.savetxt("rewards", rewardArr)
