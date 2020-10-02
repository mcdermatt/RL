import numpy as np
import cv2
from matplotlib import pyplot as plt


class road:

	def __init__(self,imgFile):
		self.map = cv2.imread(mapFile)
		scale_percent = 100 # percent of original size
		width = int(self.map.shape[1] * scale_percent / 100)
		height = int(self.map.shape[0] * scale_percent / 100)
		dim = (width, height)
		# resize image
		self.map = cv2.resize(self.map, dim, interpolation = cv2.INTER_AREA)

	def draw_map(self):

		plt.imshow(self.map, cmap = 'gray', interpolation = 'bicubic')
		plt.pause(5)
		plt.draw()

if __name__ == "__main__":

	mapFile = "track1.png"
	map = road(mapFile)
	map.draw_map()

	print("ass is ass")