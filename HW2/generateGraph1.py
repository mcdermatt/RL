import numpy as np
from numpy import convolve
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from skrrt import road


# generates plot of rewards over time

def movingAverage(values,window):
	weights = np.repeat(1.0,window)/window
	sma = np.convolve(values,weights,'valid')
	return sma

if __name__ == "__main__":

	mapFile = "track1.png"
	mapSize = 30
	Map = road(mapFile, mapSize)
	numRuns = 500
	run = 0
	vis = False
	eps = 0.01

	rew = []
	while run < numRuns:
		print("Run #",run)

		Map.evaluate(eps = eps, visual = vis) 
		rew = np.append(rew, Map.reward)

		run += 1

	win = 10
	rew = movingAverage(rew,win)

	# print(rew)
	fig2 = plt.figure(2)
	ax2 = fig2.add_subplot()
	plt.xlabel('step')
	plt.ylabel('reward')
	plt.ylim(-500,0)
	plt.title("Track 1 without Wind")

	ax2.plot(range(numRuns-win+1),rew,'b')
	plt.show()