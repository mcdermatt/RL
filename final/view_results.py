import matplotlib.pyplot as plt
import numpy as np 

def movingAverage(values,window):
	weights = np.repeat(1.0,window)/window
	sma = np.convolve(values,weights,'valid')
	return sma

fig = plt.figure(0)


# rews = np.load("critic_loss.npy")
# plt.ylim(ymax = 5e-2, ymin = -5e-2)


# rews = np.load("rewards.npy")
# plt.ylim(ymax = 1e-5, ymin = -1e-5)


rews = np.load("actor_loss.npy")
plt.ylim(ymax = 5e-2, ymin = -5e-2)


avg = movingAverage(rews,50)

# plt.ylim(ymax = 5e-2, ymin = -5e-2)

plt.plot(np.arange(1, len(rews) + 1), rews)
plt.plot(np.arange(1, len(avg) + 1), avg)

# plt.ylabel('Rewards')
# plt.ylabel('Critic Loss')
plt.ylabel('Actor Loss')
plt.xlabel('Episode #')

# plt.savefig(filename)
plt.show()