import matplotlib.pyplot as plt
import numpy as np 

def movingAverage(values,window):
	weights = np.repeat(1.0,window)/window
	sma = np.convolve(values,weights,'valid')
	return sma

#REWARDS
fig = plt.figure(0)
rews = np.loadtxt("rewards")
plt.ylabel('final distance from goal')
rews = movingAverage(rews,50)
# plt.plot(np.arange(1, len(rews) + 1), rews)
plt.ylim((-10,1))
plt.plot(np.arange(1, len(rews) + 1), rews)
plt.xlabel('Trial #')
# plt.savefig(filename)


#CRITIC LOSS
fig = plt.figure(1)
closs = np.load("critic_loss.npy")
# print(sum(closs))
plt.ylabel("critic_loss")
closs = movingAverage(closs,500)
# i = 10000
# i = 1
# while closs[i] != 0:
# 	i += 1
plt.plot(np.arange(1,len(closs)+1), np.log(closs))
# plt.plot(np.arange(1,len(closs[:i])+1), closs[:i])
plt.xlabel("Step #")

#ACTOR LOSS
fig = plt.figure(2)
aloss = np.load("actor_loss.npy")
# print(sum(aloss))
plt.ylabel("actor_loss")
aloss = movingAverage(aloss,500)
i = 10000
# i = 1
# while closs[i] != 0:
# 	i += 1
plt.plot(np.arange(1,len(aloss)+1), np.log(aloss))
# plt.plot(np.arange(1,len(aloss)+1), aloss)
# plt.plot(np.arange(1,len(closs[:i])+1), closs[:i])
plt.xlabel("Step #")

plt.show()

