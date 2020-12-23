import matplotlib.pyplot as plt
import numpy as np 

def movingAverage(values,window):
	weights = np.repeat(1.0,window)/window
	sma = np.convolve(values,weights,'valid')
	return sma

#REWARDS
fig = plt.figure(0)
rews = np.loadtxt("rewards2")
plt.ylabel('final distance from goal')
rews = movingAverage(rews,50)
# plt.plot(np.arange(1, len(rews) + 1), rews)
plt.plot(np.arange(1, len(rews) + 1), rews)
plt.xlabel('Trial #')
# plt.savefig(filename)


#CRITIC LOSS
fig = plt.figure(1)
closs = np.load("critic_loss2.npy")
# print(sum(closs))
plt.ylabel("critic_loss")
closs = movingAverage(closs,50)
i = 10000
# i = 1
# while closs[i] != 0:
# 	i += 1
plt.plot(np.arange(1,len(closs)+1), closs)
# plt.plot(np.arange(1,len(closs[:i])+1), closs[:i])
plt.xlabel("Step #")

#ACTOR LOSS
fig = plt.figure(2)
aloss = np.load("actor_loss2.npy")
# print(sum(aloss))
plt.ylabel("actor_loss")
aloss = movingAverage(aloss,50)
i = 10000
# i = 1
# while closs[i] != 0:
# 	i += 1
plt.plot(np.arange(1,len(aloss)+1), aloss)
# plt.plot(np.arange(1,len(closs[:i])+1), closs[:i])
plt.xlabel("Step #")

plt.show()

