from ragdoll import ragdoll
import numpy as np

#TODO:
# check for clipping through ground plane
# figure out what states to consider for policy
# Reward Shaping- figure out best way(?)
# change torques input to policy input
		#rather than function of timestep, torques should be function of states
# read pixels to pong
# implement Deep RL for second part of experiment
# add hip angle param
# fix rounding floor() in get_states()
# add butt velocity!!!

viz = True
arms = False
playBackSpeed = 100 #0.1
numTrials = 5000
# pol = np.load("randomPolicy2.npy")
pol = None
maxReward = 0
eps = 0.9
decay = 0.999
min_epsilon = 0.05

body = ragdoll(viz = viz, arms = arms, playBackSpeed = playBackSpeed)
body.run()
q = body.q

for trial in range(numTrials):
	print("trial number ", trial)
	torques = np.random.randn(5,500)
	body = ragdoll(pol = pol, viz = viz, arms = arms, playBackSpeed = playBackSpeed, eps = max(min_epsilon,eps*decay))
	body.run()
	pol = body.pol


body = ragdoll(pol = pol, viz = True, arms = arms, playBackSpeed = 1, eps = 0)
body.run()

np.save("randomPolicy3",pol)