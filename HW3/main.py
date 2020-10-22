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

#consider moving init policy to ragdoll 
pStep = 3
vStep = 3
pol = np.random.rand(pStep, pStep, pStep, pStep, pStep, vStep, vStep, vStep, vStep, vStep, 5)
# [rkp, lkp, rhp, lhp, bp, rkv, lkv, rhv, lhv, bv, joint actions]
pol[pol < 0.33] = -1
pol[(pol < 0.66) & (pol > 0.33)] = 0
pol[pol > 0.66] = 1


for trial in range(numTrials):
	print("trial number ", trial)
	torques = np.random.randn(5,500)
	body = ragdoll(pol,viz = viz, arms = False, torques = torques, playBackSpeed = playBackSpeed)
	body.run()