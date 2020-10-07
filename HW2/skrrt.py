import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

#TODO:
# make arrows to display policy
# implement initial random policy
# backup w/ discount factor
# draw car sprite
# sense if car has left track
# save good policy to external file
# add noise
# add seed for random initial policy
 #TODO- randomize pi via epsilon-greedy

 #DEBUG:
 #fix .onStart[] -
 #    point is being added to onStart wherever the car leaves the track

class road:

	mapSize = 30

	def __init__(self,mapFile,mapSize = mapSize, displayOn = True):
		
		#set up plot
		self.fig = plt.figure(0)
		self.ax = self.fig.add_subplot()

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

		#get args of points on road
		self.onRoad = np.argwhere(self.gw == 0) 
		#get args of points not on road
		self.offRoad = np.argwhere(self.gw == 255)
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

		#TODO fix this
		#init expected return - give everything really bad values to start?
		self.q_pi = -10000 * np.ones([mapSize,mapSize,5,5,3,3,2]) #(posx,posy,vx,vy,accelerationX, accelerationY, average reward & number of times at state so far)
		self.q_pi[:,:,:,:,:,:,0] = 0 #DEBUG -not sure if these are right
		self.q_pi[:,:,:,:,:,:,1] = 1

		#init starting policy
		np.random.seed(4)
		self.pi = np.random.rand(mapSize,mapSize,5,5,2)
		self.pi[self.pi < 0.33] = -1
		self.pi[(self.pi < 0.66) & (self.pi > 0.33)] = 0 
		self.pi[self.pi > 0.66] = 1

		self.pos = np.zeros(2)
		# self.pol = plt.arrow(0,0,0,0)
		# self.patches = []

		self.restart()

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

	def draw_policy(self):
		"""draw policy for current speeds"""

		self.ax.patches = []

		# for i in self.onRoad:
		for i in np.append(self.onRoad,self.onStart,axis = 0):
			#draw arrow (x, y, dx, dy, **kwargs)
			arrow = mpatches.Arrow(i[1] * 1000/ self.mapSize, i[0] * 1000/ self.mapSize ,self.pi[i[1],i[0],self.vx,self.vy,0] * 500/ self.mapSize ,self.pi[i[1],i[0],self.vx,self.vy,1] * 500/ self.mapSize, width = 30)
			
			# arrow = mpatches.FancyArrowPatch((i[1] * 1000/ self.mapSize ,i[0] * 1000/ self.mapSize), (self.pi[i[1],i[0],self.vx,self.vy,0] * 500/ self.mapSize ,self.pi[i[1],i[0],self.vx,self.vy,1] * 500/ self.mapSize))
			self.ax.add_patch(arrow)

	def evaluate(self,eps = 0.1, visual = False):
		'''evaluation step of policy improvement'''

		self.reward = 0
		self.history = np.zeros([1,6]) #(posx, posy, vx, vy, ax, ay)
		runLen = 100
		step = 0
		fin = False
		while fin == False:

			randy = np.random.rand()
			#greedy
			if randy > eps:
				ax = self.pi[self.pos[1], self.pos[0], self.vx, self.vy, 0]
				ay = self.pi[self.pos[1], self.pos[0], self.vx, self.vy, 1]
				self.vx = int(self.vx + ax)
				self.vy = int(self.vy + ay)
			#exploratory
			if randy < eps:
				ax = int(np.random.randint(3) - 1) #need to be careful with how I index these the (-1) could get a lil sus
				ay = int(np.random.randint(3) - 1)
				self.vx = self.vx + ax
				self.vy = self.vy + ay
				#still need to record in history what action was taken

			#saturate velocity
			if self.vx > 4:
				self.vx = 4
			if self.vx < 0:
				self.vx = 0
			if self.vy > 0:
				self.vy = 0
			if self.vy < -5:
				self.vy = -5

			#check if car has left boundary of track
			for i in self.offRoad:
				if np.all(self.pos == i):
					self.restart()
			if (self.pos[0] >= self.mapSize) or (self.pos[0] < 0): #beyond boundaries of map
				self.restart()
			if (self.pos[1] >= self.mapSize) or (self.pos[1] < 0): #beyond boundaries of map
				self.restart()

			self.pos[1] = self.pos[1] + self.vx
			self.pos[0] = self.pos[0] + self.vy 

			self.history = np.append(np.array([[self.pos[1],self.pos[0],self.vx,self.vy,ax,ay]]),self.history, axis = 0)
			
			#check if car is stuck
			if (step > 3):
				if np.array_equal(self.history[1],self.history[4]):
					# self.reward -= 10
					self.restart()

			#punish by 1 for each step until end is reached
			self.reward -= 1

			#stop if running for too long - Policy will never reach finish(?)
			if step > runLen:
				# print("step limit hit")
				break

			#stop if car reaches finish line - move to end of loop
			for i in self.onFinish:
				if np.all(self.pos == i):
					print("Reached Finish! Value = ", self.reward)
					self.restart()
					fin = True

			if visual:
				self.draw_policy()
				car, = plt.plot(self.pos[1] * 1000/ mapSize, self.pos[0] * 1000 / mapSize,'bo')
				self.update()
				car.remove()
			step += 1


		#estimate q_pi(s,a) -> expected return from policy pi of action a at state s
		for h in self.history:
			#TODO- make this less absurd looking
			#update average value of state-action pair
			self.q_pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),int(self.pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),0]) + 1,int(self.pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),0]) + 1,0] = (self.q_pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),int(self.pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),0]) + 1,int(self.pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),0]) + 1,0] + self.reward) / self.q_pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),int(self.pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),0]) + 1,int(self.pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),0]) + 1,1]
			#increment count for number of times state has been reached
			self.q_pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),int(self.pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),0]) + 1,int(self.pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),0]) + 1,1] += 1 

	def improve(self):
		'''improvement step of policy improvement'''

		#average returns observed after state is visited- should converge to expected value
		#use first-visit MC

		#pi -> greedy(Q)
		#pi = argmax(q_pi(s,a))
		

		# print(self.q_pi[15,15,0,0,:,:,0])

		xpos = 0
		ypos = 0
		vx = 0
		vy = 0
		while xpos < self.mapSize:
			while ypos < self.mapSize:
				while vx < 5:
					while vy < 5:
						#set policy x to whatever acceleration value in q_pi() produces highest argument in
							# print(self.q_pi[xpos,ypos,vx,vy,:,:,0])
							# returns 3x3 array for all combos of x and y

						#TODO - make this defalut to pi if other values have not been explored yet (I think this is why I was stuck at all zeros)
						self.pi[xpos,ypos,vx,vy,0] = np.unravel_index(np.argmax(self.q_pi[xpos,ypos,vx,vy,:,:,0]),self.q_pi[xpos,ypos,vx,vy,:,:,0].shape)[1] - 1
						#set y
						self.pi[xpos,ypos,vx,vy,1] = np.unravel_index(np.argmax(self.q_pi[xpos,ypos,vx,vy,:,:,0]),self.q_pi[xpos,ypos,vx,vy,:,:,0].shape)[0] - 1
						vy += 1

					vy = 0
					vx += 1
				vx = 0
				ypos += 1
			ypos = 0 
			xpos += 1

		return

	def draw_car(self):
		#make temporary array to hold car sprite, (add car sprite to zeros)
		# cartemp = np.zeros([1000,1000,3])
		# cartemp[self.pos[1]:self.pos[1]+100,self.pos[0]:self.pos[0]+100] = self.sprite
		plt.imshow(self.sprite)

	def restart(self):
		"""go back to start"""
		# print("Restarting")
		self.vx = 0
		self.vy = 0
		# print(np.shape(self.onStart))
		self.pos = self.onStart[np.random.randint(0,len(self.onStart))]

	def update(self):
		self.draw_map()
		# self.draw_car()	
		plt.pause(0.01)
		plt.draw()

if __name__ == "__main__":

	mapFile = "track1.png"
	mapSize = 30
	Map = road(mapFile, mapSize)
	numRuns = 10
	run = 0
	vis = False
	eps = 0.1

	while run < numRuns:
		print("Run #",run)
		Map.evaluate(eps = eps, visual = vis) 
		Map.improve()
		print(Map.onStart)
		run += 1

	Map.evaluate(eps = eps, visual = True)

	# np.savetxt('pi.txt',Map.pi[:,:,0,0,1],fmt='%.0e')