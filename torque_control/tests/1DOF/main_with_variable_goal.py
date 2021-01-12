from statePredictor import statePredictor
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
from agent import Agent
from OUNoise import OUNoise
from torch.utils.tensorboard import SummaryWriter #to print to tensorboard


#init hyperparameter search
# batch_sizes = [128, 512]
batch_sizes = [2048]
learning_rates = [0.001]
# discount_factor = [0.99]

fidelity = 0.01 #0.01 # seconds per step
trials = 25000
doneThresh = 0.1 #stop trial of theta gets within this distance with low velocity
maxTrialLen = 250
action_scale = 3 #0.01 #3
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

noise = OUNoise(1)
sp = statePredictor()
sp.dt = fidelity
sp.numPts = 2
# sp.numerical_constants[5:] = 0 #disable friction
# sp.numerical_constants[4] = 0 #no gravity


for batch_size in batch_sizes:
	for learning_rate in learning_rates:
		writer = SummaryWriter(f'runs/test4/BatchSize {batch_size} LR {learning_rate} Nodes 16-16-16-16 Tau 0.01 DF 0.9 NO_DONE LINEAR_REWARD')

		agent = Agent(3,1, LR_ACTOR=learning_rate, LR_CRITIC=learning_rate, BATCH_SIZE=batch_size)
		#init arrays for tracking results
		count = 0
		rewardArr = np.zeros(1)

		actor_loss = np.zeros((trials+1)*maxTrialLen)
		critic_loss = np.zeros((trials+1)*maxTrialLen)
		action_vec = np.zeros((trials+1)*maxTrialLen)
		tick = 0

		for trial in range(trials):
			noise.reset()
			print("trial ", trial, " batch size: ", batch_size, "learning_rate: ", learning_rate, " ---------------------------------")
			#get initial states
			goal_pos = torch.randn(1)
			# goal_pos = torch.zeros(1) #easy mode
			states = torch.randn(2)
			# states = torch.zeros(2)
			next_states = states

			tick = 0
			done = 0
			while done != 1:

				states = next_states.float()
				states = states.to(device)

				agent.actor.eval() #decide on action given states
				with torch.no_grad():
					action = agent.actor(torch.cat((states,goal_pos), dim=0).unsqueeze(0))
				agent.actor.train()

				action = noise.get_action(action.cpu().detach().numpy()[0])

				sp.numerical_specified[0] = action*action_scale
				action = torch.from_numpy(action)
				sp.x0 = states

				next_states = sp.predict()[1]
				next_states = torch.as_tensor(next_states)
				states = torch.as_tensor(states)
				# reward = -(abs(states[0] - goal_pos)**2)
				# reward = -(abs(states[0] - goal_pos)**0.5)
				reward = -(abs(states[0] - goal_pos))
				reward = torch.as_tensor(reward)
				dist = (abs(states[0] - goal_pos))

				if tick == (maxTrialLen-1):
					# reward -= 10 #punishment for not finihsing
					done = 1
				# if (abs(sp.x0[0] - goal_pos)) < doneThresh and (abs(sp.x0[1]) < 0.1): #actual goal for 1dof -> go to this position and stop
					# done = 1
				done = torch.as_tensor(done)

				if done == 0:
					agent.step(torch.cat((states,goal_pos), dim=0).cpu().numpy(), action.cpu().numpy(), reward.cpu().numpy(), torch.cat((next_states,goal_pos), dim=0).cpu().numpy(), done.cpu().numpy())
				
				# print("states = ",states)
				# print("next_states = ", next_states)

				tick += 1
				actor_loss[count] = agent.aLossOut #straight from critic
				critic_loss[count] = agent.cLossOut	
				writer.add_scalar('Actor Loss',agent.aLossOut, global_step = count)
				writer.add_scalar('Critic Loss',agent.cLossOut, global_step = count)
				writer.add_scalar('Reward',dist, global_step = count)
				count += 1	

			print("goal = ", goal_pos, " states = ",states, " action = ", action.cpu().detach().numpy()[0])
			
			rewardArr = np.append(rewardArr, reward.cpu().numpy())

			if trial % 10 == 0:
				if save_progress:
					agent.save_models()
					np.save("actor_loss",actor_loss)
					np.save("critic_loss",critic_loss)
					np.savetxt("rewards", rewardArr)
