import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from time import sleep
from statePredictor import statePredictor
from sweep import sweep

dof = 3
fidelity = 3
numBandits = fidelity ** dof
numTrials = 10 #trials per epoch
epochs = 100
dt = 0.1 #0.5 #time between start and stop of each trial
# scale = np.array([0.25,0.25,0.125]) #largest likely value for each parameter (static, kinetic, viscous respectively)
scale = 0.25
frac_resample = 0.2 #resample this fraction of worst performing bandits
eps = 0.1 #random exploration constant
sampleNoise = 0.001 #noise added when sampleing
actNoise = 0.0001 #noise added to all actions

gt = statePredictor() #ground truth model
gt.numPts = 2
gt.dt = dt
ef = statePredictor() #estimated friction model
ef.numPts = 2
ef.dt = dt

#init bandits
bandits = sweep(dof,fidelity)*scale
high_score = 0
high_score_bandit = np.zeros(3)

for epoch in range(epochs):
	print("Epoch ", epoch)
	# print(bandits)
	rew = np.zeros([numBandits,numTrials])
	avgRew = np.zeros([numBandits,numTrials])

	#randomization
	for ban in range(numBandits):
		rando = np.random.rand()
		if rando > eps:
			bandits[ban] = np.random.rand(dof)*scale

	for trial in range(numTrials):
		#get random starting states
		# print("trial ", trial+1, "/", numTrials)
		start = np.random.randn(2)
		gt.x0 = start
		ef.x0 = start

		for ban in range(numBandits):
			#add action noise
			bandits[ban] = bandits[ban]*(1 + np.random.randn(dof)*actNoise)
			ef.numerical_constants[-dof:] = bandits[ban]
			result = ef.predict()[-1]
			truth = gt.predict()[-1]
			error = np.e**(-abs(result[1]-truth[1]))
			rew[ban,trial] = error
			avgRew[ban,trial] = np.mean(rew[ban,:trial+1])

	# print(avgRew[:,-1])
	best = np.argmax(avgRew[:,-1])
	# print("best score is ", max(avgRew[:,-1]))
	# print(bandits[best])
	if max(avgRew[:,-1]) > high_score:
		print("new high score is ", max(avgRew[:,-1]))
		high_score = max(avgRew[:,-1])
		high_score_bandit = bandits[best]
		print("by ", high_score_bandit)

	#resample bandits
	#get lower quantile of poorly performing samples
	thresh = np.quantile(avgRew[:,-1],frac_resample,0) 
	#generate roulette wheel for random sampling proportional to score
	better_actions = np.argwhere(avgRew[:,-1] > thresh)
	# print("ba ", better_actions)
	action_total = np.sum(avgRew[better_actions,-1])
	wheel = np.zeros(len(better_actions))
	for i in range(len(better_actions)):
		wheel[i] = np.sum(better_actions[:i])

	for i in range(numBandits):
		if avgRew[i,-1] < thresh:
			#randomly replace near better point
			rand = np.random.rand()*action_total
			for j in range(len(wheel)):
				if rand > wheel[j]:
					bandits[i] = bandits[better_actions[j]] + np.random.randn()*sampleNoise
					break

np.savetxt('bandit_vals.txt', bandits)
print("best score is ", high_score)
print("by ", high_score_bandit)

	#randomly select for exploration