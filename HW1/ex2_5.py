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
# alpha = 1/n -> "sample-agerage method"

#QUESTIONS:
#how much should each reward be? Is it ok to have each be 1 and just the prob of selection changes?

#TODO:
#expand ban to add different results when accounting for different type of alpha

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from time import sleep

if __name__ == "__main__":

	fig = plt.figure(0)
	ax1 = fig.add_subplot(211)
	ax1.set_xlabel('Steps')
	ax1.set_ylabel('Average Reward')
	ax2 = fig.add_subplot(212)
	ax2.set_xlabel('Steps')
	ax2.set_ylabel('% Optimal Action')

	# init bandits
	numBandits = 10
	initialEst = 0.25 #higher value will make greedy algo search more
	eps = 0.25 #set epsilon param
	stepSizeWA = 0.01 #weighted average step size
	walkDist = 0.01
	# walkDist = 0
	banHistory = initialEst*np.ones([numBandits,1])

	ban = np.zeros([numBandits,4])
	ban[:,0] = 0.1*np.ones(numBandits) #set the true initial value of each bandit
	ban[:,1] = initialEst #set the initial estimate of each bandit for SA case
	ban[:,3] = initialEst #set the initial estimate of each bandit for WA case

	runLen = 500
	step = 1
	while step < runLen:

		rand = np.random.rand()
		#Greedy
		if rand > eps:	
			#get arg(s) of bandits that have highest ESTIMATE of return
			best = np.argwhere(ban[:,1] == np.amax(ban[:,1]))
			
			#case of only one best est
			if len(best) == 1:
				choice = best[0][0]
			
			#if more than one best est, pick one at radom
			else:
				choice = best[np.random.randint(len(best))][0]

			color = 'b'


		#Not Greedy -pick random
		else:
			choice = np.random.randint(numBandits)
			# color = 'r'
			color = 'w'

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
			ban[choice,3] = R #do same for WA
			ban[choice,2] = 1
			stepSizeSA = 1 # make the stepsize val 1
		#bandit has been picked already
		else:
			stepSizeSA = 1/(ban[choice,2])
			ban[choice,1] = ban[choice,1] + stepSizeSA*(R - ban[choice,1]) #update estimate of bandit reward for SA
			ban[choice,3] = ban[choice,3] + stepSizeWA*(R - ban[choice,3]) #update estimate of reward for WA
			ban[choice,2] += 1 #update number of times bandit has been picked



		#walk prob of success for each bandit
		ban[:,0] = ban[:,0] + walkDist*np.random.randn(numBandits)
		#make sure no probability is less than 0
		ban[ban[:,0] < 0] = 0
		#make sure probability of success is never more than 1 by setting soft cap on probability
		ban[ban[:,0] > 0.5,0] = ban[ban[:,0] > 0.5,0] - ban[ban[:,0] > 0.5,0]**4
		# ban[:,0] = ban[:,0] - ban[:,0] ** 3

		#update bandit history
		banHistory = np.append(banHistory,ban[:,0].reshape(-1,1),axis=1)

		#print actual values of each bandit

		#make metric of average success

		#plot average results


		#plot weighted average vs fixed average and how each responds to both stationary and moving bandits
		#hypothesis: weighted average (aka constant step size) should work better for moving value problems

		#draw points
		if step % 10 == 0:
			SA, = ax1.plot(step,ban[choice,1],color+'.') #Sample-Average Method red pts are exploration, blue are greedy
			WA, = ax1.plot(step,ban[choice,3],color+'x') #Weighted-Average Method
			# BB, = ax1.plot(step,np.max(ban[:,0]),'.k') #best probability

			# AB, = ax1.plot(step*np.ones(numBandits),ban[:,0],marker = '.', color = [0, 1, 0]) #all bandits as just points for one step

			#all bandits as points for all steps
			# for a in np.arange(0,numBandits):
			# 	ax1.plot(step,ban[a,0],color = [a/numBandits, 1 - a/numBandits, a/numBandits], marker = '.')

			
			# ax1.plot(banHistory)
			for a in np.arange(0,numBandits):
			# 	# print(np.arange(0,step+1))
			# 	# print(banHistory[:,a])
				banData = np.array([np.arange(0,step+1),banHistory[a,:]])
				ax1.plot(banData[0],banData[1], lw = 0.01)
			plt.draw()
			plt.pause(0.01)

			# AB.remove()


		print('step = ', step)
		# print(ban)
		# print('best prob = ', ban[choice,1])
		# print(banHistory)
		step += 1

	print(banHistory)
	sleep(5)