import numpy as np
from numpy import convolve
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
# from skrrt import road # for track 1
from skrrtPart2 import road #for track 5

fig = plt.figure(0)
ax = fig.add_subplot()
ax.patches = []

# mapFile = "track1.png"
mapFile = "track5.png"
mapSize = 30
Map = road(mapFile, mapSize = mapSize, wind = 0)

#use pretrained policy pi
# pi = np.load('pi1_v2_no_wind.npy')
# Map.pi = pi

numRuns = 1024
run = 0
vis = False
eps = 0.2

pic = cv2.imread(mapFile)
scale_percent = 100 # percent of original size
width = int(pic.shape[1] * scale_percent / 100)
height = int(pic.shape[0] * scale_percent / 100)
dim = (width, height)
pic =cv2.resize(pic, dim, interpolation = cv2.INTER_AREA)
plt.imshow(pic, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])

# init struct to hold best solutions for different starting positions
best = []
for i in range(np.shape(Map.onStart)[0]):
	best.append([])
#start with arbitrarily poor result
vals = -5000*np.ones(len(best))

while run < numRuns:

		if run == numRuns / 2:
			eps = eps / 2
		if run == numRuns / 4:
			eps = eps / 2
		if  run == numRuns / 8:
			eps = eps / 2

		print("Run #",run)
		Map.evaluate(eps = eps, visual = vis) 
		val = Map.reward
		count = 0
		for i in Map.onStart:
			#TODO find last start point trial was at (lots of restarting for some pts) and chop off everything after
			#find which point trial started on
			# if np.all(np.flip(Map.history[-2,:2]) == i):
			size = np.shape(Map.history[:,:2])[0]
			arg = 0
			while arg < size:
			# for arg in range(np.shape(Map.history[:,:2])[0]):
				if np.all(np.flip(Map.history[arg,:2]) == i):	
					#chop of end
					Map.history = Map.history[:arg,:]
					size = np.shape(Map.history[:,:2])[0]
					print("chop")
					if val > vals[count]:	
						vals[count] = val
						best[count] = Map.history
				arg += 1
			count += 1

		print(vals)
		run += 1

# print(best[np.argmax(vals)])
# print(best[0])
# print(len(best[4]))
n = 0
for i in best:
	# try:
	if vals[n] > -50:
		#plot path
		plt.plot(i[:(len(i)-1),0]*1000/mapSize,i[:(len(i)-1),1]*1000/mapSize)

		#plot arrows
		# print(i)
		for step in i:

			# print(step[1], step[0])
			# draw arrow (x, y, dx, dy, **kwargs)
			arrow = mpatches.Arrow(step[0] * 1000/ Map.mapSize, step[1] * 1000/ Map.mapSize, step[4] * 500/ Map.mapSize , step[5] * 500/ Map.mapSize, width = 30)
			ax.add_patch(arrow)

	# except:
	# 	pass

	n += 1


plt.pause(10)
plt.draw()
plt.savefig("Track_5_SolnV10.png")
