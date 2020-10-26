import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


class road:

	mapSize = 30

	def __init__(self,mapFile,mapSize = mapSize, displayOn = True, wind = 0.1):
		
		#set up plot
		self.fig = plt.figure(0)
		self.ax = self.fig.add_subplot()

		#set up base map image
		self.mapSize = mapSize
		self.wind = wind
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
		self.q_pi = -50000 * np.ones([mapSize,mapSize,9,9,3,3,2]) #(posx,posy,vx,vy,accelerationX, accelerationY, average reward & number of times at state so far)
		self.q_pi[:,:,:,:,:,:,1] = 1 #set initial counts to 1

		#init starting policy
		# np.random.seed(4)
		self.pi = np.random.rand(mapSize,mapSize,5,5,2)
		self.pi[self.pi < 0.33] = -1
		self.pi[(self.pi < 0.66) & (self.pi > 0.33)] = 0 
		self.pi[self.pi > 0.66] = 1

		self.pos = np.zeros(2)

		self.restart()

		self.test = 0
		# self.history = np.zeros([1,6])
		self.history = np.zeros([1,7])
		self.discountFactor = 0.1

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

		self.test += 1
		# print("test var = ", self.test)

		self.reward = 0
		self.history = np.zeros([1,7]) #(posx, posy, vx, vy, ax, ay, isEnd)
		runLen = 500
		step = 0
		fin = False
		while fin == False:

			axApparent = 0
			ayApparent = 0
			randy = np.random.rand()
			#greedy
			if randy > eps:
				axApparent = self.pi[self.pos[1], self.pos[0], self.vx, self.vy, 0]
				ayApparent = self.pi[self.pos[1], self.pos[0], self.vx, self.vy, 1]
			#exploratory
			if randy < eps:
				axApparent = int(np.random.randint(3) - 1) #need to be careful with how I index these the (-1) could get a lil sus
				ayApparent = int(np.random.randint(3) - 1)
				
			windy = np.random.rand()
			if windy > (1- self.wind):
				ax = 0
				ay = 0
			else:
				ax = axApparent
				ay = ayApparent

			self.vx = int(self.vx + ax)
			self.vy = int(self.vy + ay)

			#saturate velocity
			if self.vx >= 5:
				self.vx = 4
			if self.vx <= -5:
				self.vx = -4
			if self.vy >= 5:
				self.vy = 4
			if self.vy <= -5:
				self.vy = -4


			self.pos[1] = self.pos[1] + self.vx
			self.pos[0] = self.pos[0] + self.vy 

			#check if car has left boundary of track
			for i in self.offRoad:
				if np.all(self.pos == i):
					# self.reward -= 50
					# self.reward = -1000
					self.restart()
					# fin = True
			if (self.pos[0] >= self.mapSize) or (self.pos[0] < 0): #beyond boundaries of map
				# self.reward -= 50
				# self.reward = -1000
				self.restart()
				# fin = True
			if (self.pos[1] >= self.mapSize) or (self.pos[1] < 0): #beyond boundaries of map
				# self.reward -= 50
				# self.reward = -1000
				self.restart()
				# fin = True


			#don't take ax ay - use from policy so that wind messes stuff up
			self.history = np.append(np.array([[self.pos[1],self.pos[0],self.vx,self.vy,axApparent,ayApparent,0]]),self.history, axis = 0)

			#check if car is stuck
			if (step > 3):
				if np.array_equal(self.history[1],self.history[4]):
					# self.reward -= 50
					# self.reward -1000
					self.restart()
					# fin = True

			#punish by 1 for each step until end is reached
			self.reward -= 1

			#stop if running for too long - Policy will never reach finish(?)
			if step > runLen:
				self.restart()
				# print("step limit hit")
				break

			#stop if car reaches finish line - move to end of loop
			for i in self.onFinish:
				if np.all(self.pos == i):
					# self.reward += 500
					print("Reached Finish! Value = ", self.reward)
					#flag step as finishing
					self.history[0,6] = 1
					self.restart()
					fin = True

			if visual:
				self.draw_policy()
				car, = plt.plot(self.pos[1] * 1000/ mapSize, self.pos[0] * 1000 / mapSize,'bo')
				self.update()
				car.remove()
			step += 1


		#Attempt #2 from RL book page 110
		#sum of discounted rewards
		G = 0.0
		#importance sampling ratio
		W = 1.0
		t = 0
		for h in self.history:
			# update reward since step t
			G = self.discountFactor * G + self.reward
			# 	#increment count for number of times state has been reached
			self.q_pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),int(h[4]),int(h[5]),1] += W 
			# self.q_pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),int(h[4]),int(h[5]),0] = (W / self.q_pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),int(h[4]),int(h[5]),1]) * (G - self.q_pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),int(h[4]),int(h[5]),0])

			# print("t = ", t, " ", self.q_pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),int(h[4]),int(h[5]),0])
			self.q_pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),int(h[4]),int(h[5]),0] = self.q_pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),int(h[4]),int(h[5]),0] + (W / self.q_pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),int(h[4]),int(h[5]),1]) * (G - self.q_pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),int(h[4]),int(h[5]),0])

			#moved from improve()
			top = np.argwhere(self.q_pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),:,:,0] == np.amax(self.q_pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),:,:,0])) #get list of all args equal to arg of highest value
			best = top[np.random.randint(len(top))]
			self.pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),0] = best[1] - 1
			self.pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),1] = best[0] - 1

			W = W * (1-eps) ** t
			t+=1

			#A != pi(St)
			if (self.pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),0] != h[4]) and (self.pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),1] != h[5]):
				break 


			# #check if behavior followed target policy action
			# if h[4] != self.pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),0]:
			# 	if h[5] != self.pi[int(h[0]),int(h[1]),int(h[2]),int(h[3]),1]:
			# 		# print("not following target policy")
			# 		break
			#Update W


		#Try #3
		#importance sampling
		#find at what points in history the agent reached the finish
		# ends = np.argwhere(Map.history[:,6] == 1)
		# G = 0.0 #sum of discount rewards of BEHAVIOR policy
		# t = 0 #number of timesteps between h and finish
		
		# for h in self.history:

		# 	G = self.discountFactor * G + self.reward
		# 	rho = (1 - eps) ** t
		# 	Tau = [] #set of all time steps in which state s is visited 
			
		# 	for i in range(np.shape(self.history[0])):
		# 		if (Map.history[i][:6] == h[:6]):
		# 			np.append(Tau,h)

		# 	#First time of termination after current step (don't forget we are counting back here)
		# 	for i in ends:
		# 		if (i < t) and (i > t+1):
		# 			T = i 


		# 	t += 1


			

	def improve(self):
		'''improvement step of policy improvement'''

		#indexes not values
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

						#Importance Sampling
						# tau = np.argwhere(self.history == self.history[xpos,ypos,vx,vy,:,:,:])		


						# pick one at random if more than one max
						top = np.argwhere(self.q_pi[xpos,ypos,vx,vy,:,:,0] == np.amax(self.q_pi[xpos,ypos,vx,vy,:,:,0])) #get list of all args equal to arg of highest value

						best = top[np.random.randint(len(top))]

						self.pi[xpos,ypos,vx,vy,0] = best[1] - 1
						self.pi[xpos,ypos,vx,vy,1] = best[0] - 1
						
						#kinda worked - 
						# self.pi[xpos,ypos,vx,vy,0] = np.unravel_index(np.argmax(self.q_pi[xpos,ypos,vx,vy,:,:,0]),self.q_pi[xpos,ypos,vx,vy,:,:,0].shape)[1] - 1
						# #set y
						# self.pi[xpos,ypos,vx,vy,1] = np.unravel_index(np.argmax(self.q_pi[xpos,ypos,vx,vy,:,:,0]),self.q_pi[xpos,ypos,vx,vy,:,:,0].shape)[0] - 1
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
		self.vx = 0
		self.vy = 0
		self.pos[:] = self.onStart[np.random.randint(0,len(self.onStart))]
		self.pos = self.pos.astype(int)

	def update(self):
		self.draw_map()
		plt.pause(0.01)
		plt.draw()

if __name__ == "__main__":

	mapFile = "track5.png"
	mapSize = 30
	Map = road(mapFile, mapSize)
	numRuns = 1000
	run = 0
	vis = False
	eps = 0.05

	while run < numRuns:
		print("Run #",run)
		Map.evaluate(eps = eps, visual = vis) 
		run += 1
		if run % 25  == 0:
			Map.evaluate(eps = eps, visual = True)

	np.save('pi5',Map.pi)