# Exercise 2.5 (programming) Design and conduct an experiment to demonstrate the
# diculties that sample-average methods have for nonstationary problems. Use a modified
# version of the 10-armed testbed in which all the q⇤(a) start out equal and then take
# independent random walks (say by adding a normally distributed increment with mean 0
# and standard deviation 0.01 to all the q⇤(a) on each step). Prepare plots like Figure 2.2
# for an action-value method using sample averages, incrementally computed, and another
# action-value method using a constant step-size parameter, ↵ = 0.1. Use " = 0.1 and
# longer runs, say of 10,000 steps.

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from time import sleep

if __name__ == "__main__":

	numBandits = 10
	initialEst = 0.5 #higher value will make greedy algo search more
	eps = 0.1 #set epsilon param
	stepSizeWA = 0.125 #weighted average step size
	walkDist = 0.0
	runLen = 10000
	numRuns = 2000
	banSTD = 1 #standard deviation of rewards within each bandit

	fig = plt.figure(0)
	SA_patch = mpatches.Patch(color = 'red', label = 'Sample-Average')
	WA_patch = mpatches.Patch(color = 'blue', label = 'Weighted-Average')

	ax1 = fig.add_subplot(211)
	ax1.set_xlabel('Steps')
	ax1.set_ylabel('Average Reward')
	ax1.set_ylim([0,2])
	ax1.legend(handles = [SA_patch, WA_patch])
	ax1.set_title('Stationary Bandits with ⍺ = %f' % stepSizeWA)

	ax2 = fig.add_subplot(212)
	ax2.set_xlabel('Steps')
	ax2.set_ylabel('% Optimal Action')
	ax2.legend(handles = [SA_patch, WA_patch])

	
	#record history of rewards at each step for every run
	wa = np.zeros([runLen,numRuns]) 
	sa = np.zeros([runLen,numRuns])

	#for each step record if method was guessing optimally
	optWA = np.zeros([runLen-1,numRuns])
	optSA = np.zeros([runLen-1,numRuns])

	run = 0
	while run < numRuns:
		# numSuccSA = 0
		# numSuccWA = 0
		# banHistory = initialEst*np.ones([numBandits,1])

		ban = np.zeros([numBandits,5])

		#set the true initial value of each bandit to be equal
		# ban[:,0] = 0.1*np.ones(numBandits) 

		#linear spaced bandits
		ban[:,0] = (np.arange(0,numBandits) / (numBandits))

		ban[:,1] = initialEst #set the initial estimate of each bandit for SA case
		ban[:,3] = initialEst #set the initial estimate of each bandit for WA case
		
		historyWA = np.zeros(runLen)
		historySA = np.zeros(runLen)

		step = 0
		while step < (runLen - 1):

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

			#Not Greedy -pick random
			else:
				choiceSA = np.random.randint(numBandits)
				choiceWA = np.random.randint(numBandits)

			#record if methods are chosing the optimal bandit
			if choiceWA == np.argmax(ban[:,0]):
				optWA[step,run] = 1
			if choiceSA == np.argmax(ban[:,0]):
				optSA[step,run] = 1
			
			RSA = banSTD*np.random.randn() + ban[choiceSA,0]

			# historySA = np.append(historySA,RSA)
			historySA[step] = RSA

			#bandit has not been picked by SA method yet
			if ban[choiceSA,2] == 0:
				ban[choiceSA,1] = RSA #set reward to whatever the roll was
				ban[choiceSA,2] = 1
				stepSizeSA = 1 # make the stepsize val 1
			#bandit has been picked already by SA method
			else:
				stepSizeSA = 1/(ban[choiceSA,2])
				ban[choiceSA,1] = ban[choiceSA,1] + stepSizeSA*(RSA - ban[choiceSA,1]) #update estimate of bandit reward for SA
				ban[choiceSA,2] += 1 #update number of times bandit has been picked

			RWA = banSTD*np.random.randn() + ban[choiceWA,0]


			ban[choiceWA,3] = ban[choiceWA,3] + stepSizeWA*(RWA - ban[choiceWA,3]) #update estimate of reward for WA
			
			historyWA[step] = RWA

			#walk prob of success for each bandit
			ban[:,0] = ban[:,0] + walkDist*np.random.randn(numBandits)
			#make sure no probability is less than 0
			# ban[ban[:,0] < 0] = 0
			#make sure probability of success is never more than 1 by setting soft cap on probability
			# ban[ban[:,0] > 0.5,0] = ban[ban[:,0] > 0.5,0] - ban[ban[:,0] > 0.5,0]**4
			#hard cap
			# ban[ban[:,0] > 0.9] = ban[ban[:,0] > 0.9] * 0.9
			step += 1
		wa[:,run] = historyWA
		sa[:,run] = historySA

		#average all nonzero outcomes and graph
		cumWA = np.mean(wa[:,:run], axis = 1)
		cumSA = np.mean(sa[:,:run], axis = 1)

		# if run % 10 == 0:
		# 	WAPlot, = ax1.plot(cumWA, color = 'b', lw = 0.5)
		# 	SAPlot, = ax1.plot(cumSA, color = 'r', lw = 0.5)
			
		# 	#plot % optimal bandit chosen by each method
		# 	WAOptPlot, = ax2.plot(100*np.mean(optWA[:,:run], axis = 1) , color = 'b', lw = 0.5)
		# 	SAOptPlot, = ax2.plot(100*np.mean(optSA[:,:run], axis = 1) , color = 'r', lw = 0.5)

		# 	plt.draw()
		# 	plt.pause(0.01)
		# 	WAPlot.remove()
		# 	SAPlot.remove()
		# 	WAOptPlot.remove()
		# 	SAOptPlot.remove()

		print('run number: ',run,' of ', numRuns)
		# print(ban)
		run += 1
	
	np.save('wa_pt125Stationary.npy',cumWA)
	np.save('saStationary.npy',cumSA)
	np.save('waOpt_pt125Stationary.npy',100*np.mean(optWA[:,:run], axis = 1))
	np.save('saOptStationary.npy',100*np.mean(optSA[:,:run], axis = 1))

	# plt.savefig('2_5_Monte_Carlo.png')
	sleep(5)