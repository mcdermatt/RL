import numpy as np
import cv2
from matplotlib import pyplot as plt


class road:

	mapSize = 10
	# mapSize = 30 #default

	def __init__(self,imgFile,mapSize = mapSize):
		#set up base map image
		self.map = cv2.imread(mapFile)
		scale_percent = 100 # percent of original size
		width = int(self.map.shape[1] * scale_percent / 100)
		height = int(self.map.shape[0] * scale_percent / 100)
		dim = (width, height)
		# resize image
		self.map = cv2.resize(self.map, dim, interpolation = cv2.INTER_AREA)

		#init grid world array
		#grid world describes rewards of each discrete region on track
		self.gw = np.zeros([mapSize, mapSize])

		#loop through x and y coords of map to get values based on colors of input image
		row = 0 
		col = 0
		while row < mapSize:
			while col < mapSize:
				self.gw[row,col] = self.map[row * int(1000 / mapSize), col * int(1000 / mapSize),2]
				col += 1
			col = 0
			row += 1

		#init states var
		self.states = np.zeros([mapSize,mapSize,5])

	def draw_map(self):
		#draw base map image
		plt.imshow(self.map, cmap = 'gray', interpolation = 'bicubic')

		#draw grid world

		#get args of start points

		#get args of end points

		row = 0
		col = 0


		plt.pause(5)
		plt.draw()

if __name__ == "__main__":

	mapFile = "track1.png"
	map = road(mapFile)
	map.draw_map()

	print(map.gw)
	print("ass is ass")