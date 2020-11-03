import matplotlib.pyplot as plt
import numpy as np 

def movingAverage(values,window):
	weights = np.repeat(1.0,window)/window
	sma = np.convolve(values,weights,'valid')
	return sma

rews = np.load("rewards.npy")
avg = movingAverage(rews,50)

fig = plt.figure()
plt.plot(np.arange(1, len(rews) + 1), rews)
plt.plot(np.arange(1, len(avg) + 1), avg)

plt.ylabel('Reward')
plt.xlabel('Episode #')
plt.show()