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
playBackSpeed = 1
numTrials = 10
pol = None

for trial in range(numTrials):
	print("trial number ", trial)
	torques = np.random.randn(5,500)
	body = ragdoll(pol = pol, viz = viz, arms = False, torques = torques, playBackSpeed = playBackSpeed)
	# body = ragdoll(viz = viz, arms = False, torques = torques, playBackSpeed = playBackSpeed)
	pol = body.pol
	body.run()
