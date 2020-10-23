from ragdoll import ragdoll
import numpy as np

#TODO:
# check for clipping through ground plane
# figure out what states to consider for policy
# Reward Shaping- figure out best way(?)
# change torques input to policy input
		#rather than function of timestep, torques should be function of states
 
# torques = np.random.randn(5,500)
viz = True
arms = False
playBackSpeed = 1
numTrials = 1000
pol = None
maxReward = 0

for trial in range(numTrials):
	print("trial number ", trial)
	torques = np.random.randn(5,500)
	# body = ragdoll(pol = pol, viz = viz, arms = arms, torques = torques, playBackSpeed = playBackSpeed)
	body = ragdoll(viz = viz, arms = arms, torques = torques, playBackSpeed = playBackSpeed)
	body.run()
	if body.reward > maxReward:
		pol = body.pol
		maxReward = body.reward

body = ragdoll(pol = pol, viz = True, arms = arms, torques = torques, playBackSpeed = 1)
body.run()

np.save("randomPolicy",pol)