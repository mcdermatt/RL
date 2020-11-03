from model import Actor, Critic
from ragdoll import ragdoll
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
from agent import Agent
import numpy


#TODO 
#	BUG FIX:
#		Critic loss decreasing, actor loss near constant- doing weird shit
#	Check the done flag in my replayBuffer

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	torch.set_default_tensor_type('torch.cuda.FloatTensor') 
	print("Running on GPU")
else:
	device = torch.device("cpu")
	torch.set_default_tensor_type('torch.FloatTensor')
	print("Running on the CPU")
torch.set_default_tensor_type('torch.cuda.FloatTensor')

#init agent (state size, action size)
agent = Agent(13,5)

# noise_mult = 0.25
# noise_decay = 0.99 # 0.99
trials = 50 #repeat simulation <trials> times
rews = numpy.zeros(trials)

for trial in range(trials):
	print("Trial #: ", trial)

	#resets simulation
	body = ragdoll(viz = True, arms = False, playBackSpeed = 20, assist = False)
	states = body.get_states()
	states = states.to(device) #send to GPU
	# while body.fallen == False:
	while body.game_over == False:
		#get actions to take from forward network
		agent.actor.eval() #lock actor
		with torch.no_grad():
			action = agent.actor.forward(states.view(-1,13))

			#add OU Noise
			# action += agent.noise.sample()

			# action += torch.randn(5) * noise_mult # add noise
			#	spinning up says I don't need fancy noise function
			# print("action (in main loop) = ",action)
		agent.actor.train() #unlock actor

		#bring back to cpu for running on model - not sure if this is necessary
		act = action.cpu().detach().numpy() + agent.noise.sample()

		#perform action determined by forward network
		body.activate_joints(act[0,0],act[0,1],act[0,2],act[0,3],act[0,4])
		# body.activate_joints(action[0,0],action[0,1],action[0,2],action[0,3],action[0,4])
		body.tick()
		reward = body.calculate_reward()
		# print("reward = ",reward)

 		#get new states after action taken
		states_next = body.get_states()
		states_next = states_next.to(device) #send to GPU

		#update agent once in a while
		agent.step(states.cpu().numpy(),action.cpu().numpy(),reward.cpu().numpy(),states_next.cpu().numpy(),int(body.game_over))
		#	if enough trials have been run, randomly samples batch of trials form memory
		#	and attempts to apply network to them

		states = states_next

		# if body.step%50 == 0:
		# 	print("action (in main loop) = ",action)

		if body.game_over == True:
			print("Reward = ", body.reward)
			print("action (in main loop) = ",action)

		# agent.step(states,action,reward,states_next,1)
	# noise_mult = noise_mult * noise_decay
	# if noise_mult < 0.05:
	# 	noise_mult = 0.05
	rews[trial] = body.reward.cpu().numpy()

numpy.save("rewards",rews)


# torch.save(agent.actor.state_dict(), 'checkpoint_actor.pth')
# torch.save(agent.critic.state_dict(), 'checkpoint_critic.pth')
# torch.save(agent.actor_target.state_dict(), 'checkpoint_actor_t.pth')
# torch.save(agent.critic_target.state_dict(), 'checkpoint_critic_t.pth')


