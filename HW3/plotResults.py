import matplotlib.pyplot as plt
import numpy as np 

def movingAverage(values,window):
	weights = np.repeat(1.0,window)/window
	sma = np.convolve(values,weights,'valid')
	return sma

# filename = "rewards.png"

# rews = np.load("critic_loss.npy")
# rews = np.load("rewards.npy")
rews = np.load("actor_loss.npy")

avg = movingAverage(rews,50)

fig = plt.figure(0)
plt.plot(np.arange(1, len(rews) + 1), rews)
plt.plot(np.arange(1, len(avg) + 1), avg)

# plt.ylabel('Rewards')
# plt.ylabel('Critic Loss')
plt.ylabel('Actor Loss')
plt.xlabel('Episode #')

# plt.savefig(filename)
plt.show()
