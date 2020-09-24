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
#update best metric to account for both SA and WA methods- !!!

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from time import sleep
from numpy import convolve

#MA filter- taken from my final project from last semester
def movingAverage(values,window):
	weights = np.repeat(1.0,window)/window
	sma = np.convolve(values,weights,'valid')
	return sma

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
	eps = 0.1 #set epsilon param
	stepSizeWA = 0.125 #weighted average step size
	historyWA = np.zeros(1)
	historySA = np.zeros(1)
	# walkDist = 0.01
	walkDist = 0
	banHistory = initialEst*np.ones([numBandits,1])
	windowSize = 100

	ban = np.zeros([numBandits,5])

	#set the true initial value of each bandit to be equal
	# ban[:,0] = 0.1*np.ones(numBandits) 
	
	#TEST- give one of the bandits an actual advantage
	# ban[0,0] = 0.4 
	
	#randomize each bandit
	# ban[:,0] = 0.5*np.random.rand(numBandits)

	#linear space bandits
	ban[:,0] = (np.arange(0,numBandits) / numBandits)* 0.25

	ban[:,1] = initialEst #set the initial estimate of each bandit for SA case
	ban[:,3] = initialEst #set the initial estimate of each bandit for WA case

	numSuccSA = 0
	numSuccWA = 0
	runLen = 1000
	step = 1
	while step < runLen:

		rand = np.random.rand()
		#Greedy
		if rand > eps:	
			#get arg(s) of bandits that have highest ESTIMATE of return
			bestSA = np.argwhere(ban[:,1] == np.amax(ban[:,1]))
			bestWA = np.argwhere(ban[:,3] == np.amax(ban[:,3]))
			
			#Start for SA
			#case of only one best est
			if len(bestSA) == 1:
				choiceSA = bestSA[0][0]
			#if more than one best est, pick one at radom
			else:
				choiceSA = bestSA[np.random.randint(len(bestSA))][0]

			#repeat for WA
			if len(bestWA) == 1:
				choiceWA = bestWA[0][0]
			#if more than one best est, pick one at radom
			else:
				choiceWA = bestWA[np.random.randint(len(bestWA))][0]

			color = 'b'


		#Not Greedy -pick random
		else:
			choiceSA = np.random.randint(numBandits)
			choiceWA = np.random.randint(numBandits)
			# color = 'r'
			color = 'w'
			print('random trial')

		#roll with probability of success according to choice bandit
		roll = np.random.rand()


		#set reward sa
		if roll < ban[choiceSA,0]: #successful roll for SA
			RSA = 1
			numSuccSA += 1
		else:
			RSA = 0
		
		#bandit has not been picked by SA method yet
		if ban[choiceSA,2] == 0:
			ban[choiceSA,1] = RSA #set reward to whatever the roll was
			ban[choiceSA,2] = 1
			stepSizeSA = 1 # make the stepsize val 1
		#bandit has been picked already by SA method
		else:
			stepSizeSA = 1/(ban[choiceSA,2])
			ban[choiceSA,1] = ban[choiceSA,1] + stepSizeSA*(RSA - ban[choiceSA,1]) #update estimate of bandit reward for SA
			# historyWA = np.append(historyWA,ban[choice,3])
			# historySA = np.append(historySA,ban[choiceSA,1])
			historySA = np.append(historySA,numSuccSA/step)

			ban[choiceSA,2] += 1 #update number of times bandit has been picked


		if roll < ban[choiceWA,0]: #successful roll for WA
			RWA = 1
			numSuccWA += 1
		else:
			RWA = 0
		
		ban[choiceWA,3] = ban[choiceWA,3] + stepSizeWA*(RWA - ban[choiceWA,3]) #update estimate of reward for WA
		# historyWA = np.append(historyWA,ban[choiceWA,3]) #store what it thinks % succss currently is 
		historyWA = np.append(historyWA,numSuccWA/step)	#store actual cumulative success

		# historyWA = np.append(historyWA,ban[choice,3])


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
		if step % 100 == 0:
			# SA, = ax1.plot(step,ban[choice,1],'bx') #Sample-Average Method red pts are exploration, blue are greedy
			# WA, = ax1.plot(step,ban[choice,3],'gx') #Weighted-Average Method
			#TODO: get rid of spikes
			# ax1.plot(np.arange(0,len(historyWA)),historyWA,'r-', lw = 0.5) 
			watemp = movingAverage(historyWA,windowSize)
			ax1.plot(np.arange(0,len(watemp))+windowSize,watemp,'g-', lw = 2) 
			satemp = movingAverage(historySA,windowSize)
			ax1.plot(np.arange(0,len(satemp))+windowSize,satemp,'b-', lw = 2) 

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


		# print('step = ', step)
		# print(ban)
		# print('best prob = ', ban[choice,1])
		# print(banHistory)
		print('choice SA + ', choiceSA, ' choiceWA = ', choiceWA)
		step += 1

	print(banHistory)
	sleep(5)