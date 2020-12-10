import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from time import sleep
from statePredictor import statePredictor

fidelity = 8 #number of discrete values to test for each parameter
numBandits = fidelity**3
numTrials = 10
dt = 0.5 #time between start and stop of each trial
scale = np.array([0.25,0.25,0.125]) #largest likely value for each parameter (static, kinetic, viscous respectively)

gt = statePredictor() #ground truth model
gt.numPts = 2
gt.dt = dt
ef = statePredictor() #estimated friction model
ef.numPts = 2
ef.dt = dt
#init bandits
# bandits = np.random.rand(numBandits,3)*scale #random

#param sweep
bandits = np.zeros([numBandits,3])
for i in range(fidelity):
	# bandits[int((numBandits/fidelity)*i):int((numBandits/fidelity)*(1+i)),0] = i / fidelity
	bandits[(fidelity**2)*i:(fidelity**2)*(1+i),0] = i / fidelity
for i in range(fidelity**2):
	bandits[i*fidelity:(i+1)*fidelity,1] = (i % fidelity) / fidelity
for i in range(fidelity**3):
	bandits[i,2] = (i % fidelity) / fidelity
bandits = bandits*scale
print(bandits)
rew = np.zeros([numBandits,numTrials])
avgRew = np.zeros([numBandits,numTrials])

#init plot
fig = plt.figure(0)
ax = fig.add_subplot()
ax.set_xlabel("trial")
ax.set_ylabel("reward")

for trial in range(numTrials):
	#get random starting states
	print("trial ", trial+1, "/", numTrials)
	start = np.random.randn(2)
	gt.x0 = start
	ef.x0 = start

	for ban in range(numBandits):
		ef.numerical_constants[5:] = bandits[ban]
		result = ef.predict()[-1]
		truth = gt.predict()[-1]
		error = np.e**(-abs(result[1]-truth[1]))
		rew[ban,trial] = error
		avgRew[ban,trial] = np.mean(rew[ban,:trial+1])

# for m in range(numBandits):
# 	ax.plot(avgRew[m])
# 	plt.draw()
# 	plt.pause(0.01)

# plt.pause(30)
print(rew[:,-1])
best = np.argmax(rew[:,-1])
print("best score is ", max(rew[:,-1]))
print(bandits[best])