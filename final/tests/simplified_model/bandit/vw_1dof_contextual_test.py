from vowpalwabbit import pyvw
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from time import sleep
from statePredictor import statePredictor
from sweep import sweep


fidelity = 10 #number of discrete values to test for each parameter
dof = 1 # (static, kinetic, viscous)
numBandits = fidelity**dof
numTrials = 10 #number of trials per epoch
dt = 0.5 #time between start and stop of each trial
scale = 0.125 #only looking at 1 param

#init simulated arms
gt = statePredictor() #ground truth model
gt.numPts = 2
gt.dt = dt
ef = statePredictor() #estimated friction model
ef.numPts = 2
ef.dt = dt

#init model
model = pyvw.vw(quiet=True)

#init bandits
bandits = sweep(dof, fidelity)*scale
print(bandits)
rew = np.zeros([numBandits,numTrials])
avgRew = np.zeros([numBandits,numTrials])

#generate test data
train_examples = []
for trial in range(numTrials):
	start = np.random.randn(2)
	gt.x0 = start
	ef.x0 = start

	for ban in range(numBandits):
		ef.numerical_constants[-1] = bandits[ban][0]#only looking at viscous
		result = ef.predict()[-1]
		truth = gt.predict()[-1]
		error = 1-np.e**(-abs(result[1]-truth[1]))
		rew[ban,trial] = error
		avgRew[ban,trial] = np.mean(rew[ban,:trial+1])

		entry = str(error) + " | " + "theta0:" + str(start[0]) +  " omega0:" + str(start[1]) #incorrect. does not take into account action taken!!!
		# entry = {"action": ban, "cost":error }
		train_examples.append(entry)
		# print(train_examples[-1])

for example in train_examples:
	model.learn(example)

test_example = "| theta0:1 omega0:-0.5"
prediction = model.predict(test_example)
print(prediction)