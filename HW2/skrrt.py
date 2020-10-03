import numpy as np
import cv2
from matplotlib import pyplot as plt

#TODO:
# make arrows to display policy
# implement initial random policy
# backup w/ discount factor
# draw car sprite
# sense if car has left track
# save good policy to external file
# add noise

class road:

	mapSize = 30

	def __init__(self,imgFile,mapSize = mapSize):
		#set up base map image
		self.mapSize = mapSize
		self.map = cv2.imread(mapFile)
		self.sprite = cv2.imread("car.png")
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
		# plt.imshow(self.sprite)
		plt.xticks([]), plt.yticks([])

		#make vx and vy display on graph
		self.vx = 0
		self.vy = 0
		self.reward = 0 #reward for current trial
		self.vxtxt = plt.text(10,75,"Vx = %i" %self.vx)
		self.vytxt = plt.text(10,125,"Vy = %i" %self.vy)
		self.rewardtxt = plt.text(700,75, "Reward = %i" %self.reward)

		#init starting policy
		self.pi = np.random.rand(mapSize,mapSize,5,5,2)
		# self.pi[self.pi < 0.33] = -1
		# self.pi[(self.pi < 0.66) & (self.pi > 0.33)] = 0 
		# self.pi[self.pi > 0.66] = 1

		self.pi[self.pi < 0.5] = -1
		self.pi[self.pi > 0.5] = 1

		self.pos = np.zeros(2)

	def draw_map(self):

		self.vxtxt.set_text("Vx = %i" %self.vx)
		self.vytxt.set_text("Vy = %i" %self.vy)
		self.rewardtxt.set_text("Reward = %i" %self.reward)
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

	def draw_car(self):
		#make temporary array to hold car sprite, (add car sprite to zeros)
		# cartemp = np.zeros([1000,1000,3])
		# cartemp[self.pos[1]:self.pos[1]+100,self.pos[0]:self.pos[0]+100] = self.sprite
		plt.imshow(self.sprite)

	def restart(self):
		"""go back to start"""
		self.vx = 0
		self.vy = 0
		self.pos = Map.onStart[np.random.randint(0,len(Map.onStart))]

	def update(self):
		self.draw_map()
		# self.draw_car()	
		plt.pause(0.1)
		plt.draw()

		#state transitions

if __name__ == "__main__":

	mapFile = "track1.png"
	mapSize = 30
	Map = road(mapFile, mapSize)
	
	#test moving car
	#start at random point in atStart
	Map.pos = Map.onStart[np.random.randint(0,len(Map.onStart))]
	# print("pos = ", pos)
	history = [] #append to this to discount rewards

	runLen = 50
	step = 0
	while step < runLen:

		Map.vx = int(Map.vx + Map.pi[Map.pos[1], Map.pos[0], Map.vx, Map.vy, 0])
		Map.vy = int(Map.vy + Map.pi[Map.pos[1], Map.pos[0], Map.vx, Map.vy, 1])

		#saturate velocity
		if Map.vx > 4:
			Map.vx = 4
		if Map.vx < 0:
			Map.vx = 0
		if Map.vy > 0:
			Map.vy = 0
		if Map.vy < -5 :
			Map.vy = -5

		vxlast = Map.vx
		vylast = Map.vy
	
		Map.pos[1] = Map.pos[1] + Map.vx
		Map.pos[0] = Map.pos[0] + Map.vy 
		car, = plt.plot(Map.pos[1] * 1000/ mapSize, Map.pos[0] * 1000 / mapSize,'bo')
		
		#check if car has left boundary of track
		for i in Map.offRoad:
			if np.all(Map.pos == i):
				print("we offRoad")
				Map.restart()
		#check if car is stuck at start

		#punish by 1 for each step until end is reached
		Map.reward -= 1

		Map.update()
		car.remove()
		step += 1
	

	np.savetxt('states.txt',Map.states[:,:,0,0],fmt='%.0e')
	# print(map.states)