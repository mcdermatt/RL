import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from time import sleep
from statePredictor import statePredictor
from sweep import sweep

#notes
#	Need way of keeping track of how much to adjust ranges of new values for each parameter between epochs
#		some values are more important than others (viscos >>> static) - 
#		if certain values tend to keep oscilating - maybe reduce the learning rate of those vals?

fidelity = 3 #number of discrete values to test for each parameter
dof = 3 # (static, kinetic, viscous)
numBandits = fidelity**dof
numTrials = 10 #number of trials per epoch
dt = 0.5 #time between start and stop of each trial
scale = np.array([0.25,0.25,0.125]) #largest likely value for each parameter (static, kinetic, viscous respectively)
scale = 0.125 #only looking at 1 param
epochs = 1

#init models
gt = statePredictor() #ground truth model
gt.numPts = 2
gt.dt = dt
ef = statePredictor() #estimated friction model
ef.numPts = 2
ef.dt = dt

bandits = sweep(dof, fidelity)*scale
# bandits = sweep(dof, fidelity)*scale/fidelity
print(bandits)

for epoch in range(epochs):
	#every loop bandits are regenerated around the area of the best bandit in the previous epoch
	print("Epoch ", epoch)
	rew = np.zeros([numBandits,numTrials])
	avgRew = np.zeros([numBandits,numTrials])

	for trial in range(numTrials):
		#get random starting states
		# print("trial ", trial+1, "/", numTrials)
		start = np.random.randn(2)
		gt.x0 = start
		ef.x0 = start

		for ban in range(numBandits):
			ef.numerical_constants[5:] = bandits[ban]
			# ef.numerical_constants[-1] = bandits[ban][0]#only looking at viscous
			result = ef.predict()[-1]
			truth = gt.predict()[-1]
			error = np.e**(-abs(result[1]-truth[1]))
			rew[ban,trial] = error
			avgRew[ban,trial] = np.mean(rew[ban,:trial+1])

	# print(rew[:,-1])
	print("best score is ", max(rew[:,-1]))
	best = np.argmax(rew[:,-1])
	print("by this bandit: ",bandits[best])

	newScale = scale * 1/(epoch+1)
	bandits = bandits[best] + newScale*(1/2 - sweep(dof, fidelity))
	bandits[bandits < 0] = 0
	# print(bandits)