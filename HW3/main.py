from ragdoll import ragdoll
import numpy as np


#TODO:
# check for clipping through ground plane
# figure out what states to consider for policy
# Reward Shaping- figure out best way(?)
# change torques input to policy input
		#rather than function of timestep, torques should be function of states
 
torques = np.random.randn(5,500)
viz = True
playBackSpeed = 100
numTrials = 10

for trial in range(numTrials):
	print("trial number ", trial)
	torques = np.random.randn(5,500)
	body = ragdoll(viz = viz, arms = True, torques = torques, playBackSpeed = playBackSpeed)
	body.run()