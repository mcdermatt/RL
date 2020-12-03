import matplotlib.pyplot as plt
import numpy as np 

def movingAverage(values,window):
	weights = np.repeat(1.0,window)/window
	sma = np.convolve(values,weights,'valid')
	return sma

fig = plt.figure(0)

rews = np.load("critic_loss.npy")
plt.ylabel('Critic Loss')
# plt.ylim(ymax = 5e-2, ymin = -5e-2)

#get last nonzero point
# lastPt = 300
# i = 1
# while (i != 0) and (lastPt < 5000):
# # while lastPt < 5000:
# 	i = rews[lastPt]
# 	lastPt += 1
lastPt = 500000
plt.xlim(xmin = 0, xmax = lastPt)

avg = movingAverage(rews,500)

# plt.plot(np.arange(1, len(rews) + 1), rews)
plt.plot(np.arange(1, len(avg) + 1), avg)

plt.xlabel('Step #')
# plt.savefig(filename)
# plt.show()

fig = plt.figure(1)
rews = np.load("rewards.npy")
plt.ylabel('Rewards')
# plt.ylim(ymax = 1e-5, ymin = -1e-5)

avg = movingAverage(rews,500)

# plt.ylim(ymax = 5e-2, ymin = -5e-2)

# plt.plot(np.arange(1, len(rews) + 1), rews)
plt.plot(np.arange(1, len(avg) + 1), avg)
plt.xlim(xmin = 0, xmax = lastPt)

plt.xlabel('Step #')

# plt.savefig(filename)
# plt.show()

fig = plt.figure(2)
rews = np.load("actor_loss.npy")
plt.ylabel('Actor Loss')
# plt.ylim(ymax = 5e-2, ymin = -5e-2)


avg = movingAverage(rews,500)

# plt.ylim(ymax = 5e-2, ymin = -5e-2)

# plt.plot(np.arange(1, len(rews) + 1), rews)
plt.plot(np.arange(1, len(avg) + 1), avg)

plt.xlabel('Step #')
plt.xlim(xmin = 0, xmax = lastPt)

# plt.savefig(filename)
plt.show()