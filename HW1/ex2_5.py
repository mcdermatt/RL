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

import numpy as np
import matplotlib
from matplotlib import pyplot as plt


if __name__ == "__main__":

	# init bandits
	numBandits = 10
	initialEst = 0.5
	eps = 0.25 #set epsilon param

	ban = np.zeros([numBandits,2])
	ban[:,0] = 0.1*np.ones(numBandits) #set the true initial value of each bandit
	ban[:,1] = initialEst #set the initial estimate of each bandit

	runLen = 100
	step = 0
	while step < runLen:



		#walk prob of success for each bandit
		ban = ban + 0.01*np.random.randn(numBandits)

		step += 1

	#print actual values of each bandit

	#plot average results

	#plot random walk of rewards for each bandit

	#plot weighted average vs fixed average and how each responds to both stationary and moving bandits
	#hypothesis: weighted average (aka constant step size) should work better for moving value problems