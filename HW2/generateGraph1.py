import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

if __name__ == "__main__":

	mapFile = "track1.png"
	mapSize = 15
	Map = road(mapFile, mapSize)
	numRuns = 500
	run = 0
	vis = False
	eps = 0.01

	while run < numRuns:
		print("Run #",run)
		# print(np.may_share_memory(Map.onStart,Map.pos))
		# print(Map.q_pi[15,15,0,0,:,:,0])

		#dynamic epsilon param
		# eps = 1 / ((run + 1)**0.5)
		Map.evaluate(eps = eps, visual = vis) 
		Map.improve()
		# if run % 10 == 0:
		# 	Map.improve()

		run += 1