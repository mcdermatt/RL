import numpy as np
import cv2
from matplotlib import pyplot as plt


class road:

	mapSize = 30

	def __init__(self,imgFile,mapSize = mapSize):
		#set up base map image
		self.mapSize = mapSize
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
		self.states = np.zeros([mapSize,mapSize,5,5])

		#get args of points on road
		self.onRoad = np.argwhere(self.gw == 0) 

		#get args of points not on road
		self.offRoad = np.argwhere(self.gw == 255)
		#set large negative reward for points off of road
		self.states[self.offRoad[:,0],self.offRoad[:,1],:] = -50

		#get args of start points
		self.onStart = np.argwhere((self.gw < 127) & (self.gw > 0)) 

		#get args of end points
		self.onFinish = np.argwhere((self.gw >= 127) & (self.gw < 255)) 

		#draw base map image
		plt.imshow(self.map, cmap = 'gray', interpolation = 'bicubic')
	
		#make vx and vy display on graph
		self.vx = 0
		self.vy = 0
		self.vxtxt = plt.text(10,75,"Vx = %i" %self.vx)
		self.vytxt = plt.text(10,125,"Vy = %i" %self.vy)

		#init starting policy
		self.pi = np.random.rand(mapSize,mapSize,5,5,2)
		self.pi[self.pi < 0.33] = -1
		self.pi[(self.pi < 0.66) & (self.pi > 0.33)] = 0
		self.pi[self.pi > 0.66] = 1
		# print(self.pi)

	def draw_map(self):

		self.vxtxt.set_text("Vx = %i" %self.vx)
		self.vytxt.set_text("Vy = %i" %self.vy)

		#draw base map image
		# plt.imshow(self.map, cmap = 'gray', interpolation = 'bicubic')		

		#draw grid world
		# plt.plot(self.onRoad[:,1]* 1000/ self.mapSize,self.onRoad[:,0]* 1000/ self.mapSize,'b.')
		# plt.plot(self.offRoad[:,1]* 1000/ self.mapSize,self.offRoad[:,0]* 1000/ self.mapSize,'r.')
		# plt.plot(self.onStart[:,1]* 1000/ self.mapSize,self.onStart[:,0]* 1000/ self.mapSize,'g.')		
		# plt.plot(self.onFinish[:,1]* 1000/ self.mapSize,self.onFinish[:,0]* 1000/ self.mapSize,'k.')
		# pass

	def evaluation(self):
		'''evaluation step of policy improvement'''

		return

	def improvement(self):
		'''improvement step of policy improvement'''

		return

	def update(self):
		self.draw_map()
		plt.pause(0.1)
		plt.draw()

		#state transitions

if __name__ == "__main__":

	mapFile = "track1.png"
	mapSize = 30
	Map = road(mapFile, mapSize)
	
	#test moving car
	#start at random point in atStart
	pos = Map.onStart[np.random.randint(0,len(Map.onStart))]
	# print("pos = ", pos)
	history = [] #append to this to discount rewards

	runLen = 15
	step = 0
	while step < runLen:

		Map.vx = int(Map.vx + Map.pi[pos[1], pos[0], Map.vx, Map.vy, 0])
		Map.vy = int(Map.vy + Map.pi[pos[1], pos[0], Map.vx, Map.vy, 1])

		#saturate velocity
		if Map.vx > 5:
			Map.vx = 0
		if Map.vx < 0:
			Map.vx = 0
		if Map.vy > 0:
			Map.vy = 0
		if Map.vy < -5 :
			Map.vy = -5

		vxlast = Map.vx
		vylast = Map.vy
	
		#TODO - do I really need pos variable? should be just using first two elements of Map?
		pos[1] = pos[1] + Map.vx
		pos[0] = pos[0] + Map.vy 
		car, = plt.plot(pos[1] * 1000/ mapSize, pos[0] * 1000 / mapSize,'bo')
		
		Map.update()
		car.remove()
		step += 1
	

	np.savetxt('states.txt',Map.states[:,:,0,0],fmt='%.0e')
	# print(map.states)
