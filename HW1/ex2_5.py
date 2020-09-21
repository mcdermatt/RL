# Exercise 2.5 (programming) Design and conduct an experiment to demonstrate the
# diculties that sample-average methods have for nonstationary problems. Use a modified
# version of the 10-armed testbed in which all the q⇤(a) start out equal and then take
# independent random walks (say by adding a normally distributed increment with mean 0
# and standard deviation 0.01 to all the q⇤(a) on each step). Prepare plots like Figure 2.2
# for an action-value method using sample averages, incrementally computed, and another
# action-value method using a constant step-size parameter, ↵ = 0.1. Use " = 0.1 and
# longer runs, say of 10,000 steps.

#NOTES:
#constant step size parameter -> weighted average (favors more recent samples)

#QUESTIONS:
#how much should each reward be? Is it ok to have each be 1 and just the prob of selection changes?

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from time import sleep

if __name__ == "__main__":

	plt.figure(0)

	# init bandits
	numBandits = 10
	initialEst = 0.5 #higher value will make greedy algo search more
	eps = 0.25 #set epsilon param
	stepSize = 0.1
	walkDist = 0.01
	# walkDist = 0

	ban = np.zeros([numBandits,3])
	ban[:,0] = 0.1*np.ones(numBandits) #set the true initial value of each bandit
	ban[:,1] = initialEst #set the initial estimate of each bandit

	runLen = 500
	step = 0
	while step < runLen:

		rand = np.random.rand()
		#Greedy
		if rand < eps:	
			#get arg(s) of bandits that have highest estimate of return
			best = np.argwhere(ban[:,1] == np.amax(ban[:,1]))
			
			#case of only one best est
			if len(best) == 1:
				choice = best[0][0]
			
			#if more than one best est, pick one at radom
			else:
				choice = best[np.random.randint(len(best))][0]

			color = 'b.'


		#Not Greedy -pick random
		else:
			choice = np.random.randint(numBandits)
			color = 'r.'

		#roll with probability of success according to choice bandit
		roll = np.random.rand()

		#set reward
		if roll < ban[choice,0]: #successful roll
			R = 1
		else:
			R = 0
		
		#bandit has not been picked yet
		if ban[choice,2] == 0:
			ban[choice,1] = R #set reward to whatever the roll was
			ban[choice,2] = 1
			stepSize = 1 # make the stepzide val 1
		#bandit has been picked already
		else:
			stepSize = 1/(ban[choice,2])
			ban[choice,1] = ban[choice,1] + stepSize*(R - ban[choice,1]) #update estimate of bandit reward
			ban[choice,2] += 1 #update number of times bandit has been picked



		#walk prob of success for each bandit
		ban[:,0] = ban[:,0] + walkDist*np.random.randn(numBandits)

		#print actual values of each bandit

		#make metric of average success

		#plot average results

		#plot random walk of rewards for each bandit

		#plot weighted average vs fixed average and how each responds to both stationary and moving bandits
		#hypothesis: weighted average (aka constant step size) should work better for moving value problems

		#draw points
		pt = plt.plot(step,ban[choice,1],color)
		plt.draw()
		plt.pause(0.01)


		print('step = ', step)
		print(ban)
		print('best prob = ', ban[choice,1])
		step += 1

	sleep(5)