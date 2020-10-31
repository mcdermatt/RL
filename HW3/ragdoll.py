import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util
import pygame
import pygame.color
from pygame.locals import *
import pickle
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from replayBuffer import ReplayBuffer

#TODO
# make step function work with viz = False
# standardize ins and outs similar to gym enviornments
# increase elasticity of collisions between feet and ground 
#	-> better simulates ankle action in running
# create ReplayBuffer class to store history
# update history within tick()
# change value func so it is accessable at every tick()

class ragdoll:
	""" torques (input) = [right knee, left knee, right hip, left hip, back] """

	def __init__(self, viz = True, arms = True, playBackSpeed = 1):

		# self.wX = 1600
		# self.wY = 800
		self.wX = 2000 #1600
		self.wY =  600
		self.startX = self.wX / 4
		self.dampingCoeff = 10000
		self.torqueMult = 10000 #50000
		self.foreground = (178,102,255,255) #foreground color
		self.midground = (153,51,255,255) 
		self.background = (127,0,255,255)
		self.floor = (96,96,96,255)
		self.sky = (32,32,32,255)
		self.fallen = False

		self.screen = pygame.display.set_mode((self.wX,self.wY))
		self.clock = pygame.time.Clock()
		self.viz = viz
		self.playBackSpeed = playBackSpeed
		# self.eps = eps
		self.discountFactor = 0.99

		if self.viz:
			#init pygame
			pymunk.pygame_util.positive_y_is_up = False
			pygame.init()			
			self.font = pygame.font.Font(None, 24)
			self.box_size = 200
			self.box_texts = {}
			self.help_txt = self.font.render(
			    "Pymunk simple human 2D demo. Use mouse to drag/drop", 
			    1, pygame.color.THECOLORS["darkgray"])
			self.mouse_joint = None
			self.mouse_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)

		#init sim space
		self.space = pymunk.Space()
		self.space.gravity = (0.0, 1200.0) #(0.0,900.0)
		self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
		self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
		self.step = 0
		self.done = False


		class Box:
		    def __init__(ragdoll, p0=(10, 10), p1=(self.wX-10,self.wY-50), d=2):
		        x0, y0 = p0
		        x1, y1 = p1
		        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
		        for i in range(4):
		            segment = pymunk.Segment(self.space.static_body, pts[i], pts[(i+1)%4], d)
		            segment.elasticity = 1
		            segment.friction = 10
		            segment.color = self.sky
		            self.space.add(segment)
		Box()

		#init body
		self.kneeMax = 0
		self.kneeMin = -2.5
		self.hipMax = 3
		self.hipMin = -1
		self.backMin = -0.25
		self.backMax = 1
		#add right leg
		self.rightShin = self.add_limb(self.space, (self.startX,500), color = self.background, friction = 25, mass = 1)
		self.rightThigh = self.add_limb(self.space, (self.startX,400), thiccness = 15, color = self.background)
		self.rightKnee = pymunk.PivotJoint(self.rightShin,self.rightThigh,(self.startX,450))
		self.rightKneeLimits = pymunk.RotaryLimitJoint(self.rightShin,self.rightThigh,self.kneeMin,self.kneeMax)
		self.rightKneeDamp = pymunk.DampedRotarySpring(self.rightShin,self.rightThigh,0,0,self.dampingCoeff)
		self.space.add(self.rightKnee)
		self.space.add(self.rightKneeLimits)
		self.space.add(self.rightKneeDamp)
		#add left leg
		self.leftShin = self.add_limb(self.space, (self.startX,500), color = self.midground, friction = 25, mass = 1)
		self.leftThigh = self.add_limb(self.space, (self.startX,400), thiccness = 15, color = self.midground)
		self.leftKnee = pymunk.PivotJoint(self.leftShin,self.leftThigh,(self.startX,450))
		self.leftKneeLimits = pymunk.RotaryLimitJoint(self.leftShin,self.leftThigh,self.kneeMin,self.kneeMax)
		self.leftKneeDamp = pymunk.DampedRotarySpring(self.leftShin,self.leftThigh,0,0,self.dampingCoeff)
		self.space.add(self.leftKneeDamp)
		self.space.add(self.leftKnee)
		self.space.add(self.leftKneeLimits)
		#add butt & hips
		buttOffset = 10 
		# buttOffset = 0
		self.butt = self.add_limb(self.space,(self.startX-buttOffset,320), length = 10, thiccness = 17,color = self.midground, COLLTYPE = 3)
		self.rightHip = pymunk.PivotJoint(self.rightThigh,self.butt,(self.startX,350))
		self.rightHipLimits = pymunk.RotaryLimitJoint(self.rightThigh,self.butt,self.hipMin,self.hipMax)
		self.rightHipDamp = pymunk.DampedRotarySpring(self.rightThigh,self.butt,0,0,self.dampingCoeff)
		self.space.add(self.rightHipDamp)
		self.space.add(self.rightHip)
		self.space.add(self.rightHipLimits)
		self.leftHip = pymunk.PivotJoint(self.leftThigh,self.butt,(self.startX,350))
		self.leftHipLimits = pymunk.RotaryLimitJoint(self.leftThigh,self.butt,self.hipMin,self.hipMax)
		self.leftHipDamp = pymunk.DampedRotarySpring(self.leftThigh,self.butt,0,0,self.dampingCoeff)
		self.space.add(self.leftHipDamp)
		self.space.add(self.leftHip)
		self.space.add(self.leftHipLimits)
		self.noSplits = pymunk.RotaryLimitJoint(self.leftThigh,self.rightThigh,-2,2)
		self.space.add(self.noSplits)
		#add back
		self.back = self.add_limb(self.space,(self.startX,245), mass = 2, length = 30, thiccness = 15, color = self.midground, COLLTYPE = 3)# filter = 0b01)
		self.spine = pymunk.PivotJoint(self.butt,self.back,(self.startX,300))
		self.spineLimits = pymunk.RotaryLimitJoint(self.butt,self.back,self.backMin,self.backMax)
		self.spineDamp = pymunk.DampedRotarySpring(self.butt,self.back,0,0,self.dampingCoeff)
		self.space.add(self.spineDamp)
		self.space.add(self.spine)
		self.space.add(self.spineLimits)
		if arms:
			#add head
			shoulderHeight = 240
			self.head = self.add_limb(self.space,(self.startX,shoulderHeight - 50), length = 10, thiccness = 22,color = self.midground, COLLTYPE = 3)
			self.neck = pymunk.PivotJoint(self.back,self.head,(self.startX, shoulderHeight - 40))
			self.neckLimits = pymunk.RotaryLimitJoint(self.back,self.head,-0.2,0.8)
			self.space.add(self.neck)
			self.space.add(self.neckLimits)
			#add right arm
			self.rightLowerArm = self.add_limb(self.space, (self.startX,shoulderHeight+80), color = self.foreground, COLLTYPE = 2)
			self.rightUpperArm = self.add_limb(self.space, (self.startX,shoulderHeight), thiccness = 12, color = self.foreground, COLLTYPE = 2)
			self.rightElbow = pymunk.PivotJoint(self.rightLowerArm,self.rightUpperArm,(self.startX,shoulderHeight+50))
			self.rightElbowLimits = pymunk.RotaryLimitJoint(self.rightLowerArm,self.rightUpperArm,0,2.2)
			self.rightElbowDamp = pymunk.DampedRotarySpring(self.rightLowerArm,self.rightUpperArm,0,0,self.dampingCoeff)
			self.space.add(self.rightElbow)
			self.space.add(self.rightElbowLimits)
			self.space.add(self.rightElbowDamp)
			#add left arm
			self.leftUpperArm = self.add_limb(self.space, (self.startX,shoulderHeight), thiccness = 12, color = self.background, COLLTYPE = 2)
			self.leftLowerArm = self.add_limb(self.space, (self.startX,shoulderHeight+80), color = self.background, COLLTYPE = 2)
			self.leftElbow = pymunk.PivotJoint(self.leftLowerArm,self.leftUpperArm,(self.startX,shoulderHeight+50))
			self.leftElbowLimits = pymunk.RotaryLimitJoint(self.leftLowerArm,self.leftUpperArm,0,2.2)
			self.leftElbowDamp = pymunk.DampedRotarySpring(self.leftLowerArm,self.leftUpperArm,0,0,self.dampingCoeff)
			self.space.add(self.leftElbow)
			self.space.add(self.leftElbowLimits)
			self.space.add(self.leftElbowDamp)
			self.rightShoulder = pymunk.PivotJoint(self.rightUpperArm,self.back,(self.startX,shoulderHeight-20))
			self.rightShoulderDamp = pymunk.DampedRotarySpring(self.rightUpperArm,self.back,0,0,self.dampingCoeff)
			self.space.add(self.rightShoulder)
			self.space.add(self.rightShoulderDamp)
			self.leftShoulder = pymunk.PivotJoint(self.leftUpperArm,self.back,(self.startX,shoulderHeight-20))
			self.leftShoulderDamp = pymunk.DampedRotarySpring(self.leftUpperArm,self.back,0,0,self.dampingCoeff)
			self.space.add(self.leftShoulder)
			self.space.add(self.leftShoulderDamp)

		# Create and add the "goal" 
		COLLTYPE_GOAL = 2
		goal_body = pymunk.Body()
		goal = pymunk.Poly(goal_body, [(10,self.wY-52),(10,self.wY),(self.wX-10,self.wY),(self.wX-10,self.wY-52)])
		goal.color = self.floor
		goal.collision_type = COLLTYPE_GOAL
		self.space.add(goal)

		COLLTYPE_BACK = 3
		self.h = self.space.add_collision_handler(COLLTYPE_BACK, COLLTYPE_GOAL)

		self.pStep = 3
		self.vStep = 3

		self.statevec = torch.zeros(13)

		#change history to replay buffer object
		self.history = ReplayBuffer(action_size = 5, buffer_size = 10000, batch_size = 100)


	def add_limb(self,space,pos,length = 30, mass = 2, thiccness = 10, color = (100,100,100,255), COLLTYPE = 1, friction = 0.7):#filter = 0b100):
		body = pymunk.Body()
		body.position = Vec2d(pos)
		shape = pymunk.Segment(body, (0,length), (0,-length), thiccness)
		shape.mass = mass
		shape.friction = friction
		shape.color = color
		# COLLTYPE_BACK = 3
		if COLLTYPE == 1:
			shape.collision_type = 1
			filter = 0b100
		if COLLTYPE == 2:
			shape.collision_type = 2
			filter = 0b110
		if COLLTYPE == 3:
			shape.collision_type = 3
			filter = 0b010 
			# filter = 0b100
		self.space.add(body, shape)
		#shapes of same filter will not collide with one another
		shape.filter = pymunk.ShapeFilter(categories=filter, mask=pymunk.ShapeFilter.ALL_MASKS ^ filter)
		return body

	#init collision callback function
	def fell_over(self,space, arbiter, x):
	    # print("player fell over at x =", self.back.position[0])
	    # pygame.quit()
	    self.fallen = True
	    self.calculate_reward()
	    # self.update_values()
	    return True

	def get_states(self):
		"""gets positions and velocities of each joint in Rad and Rad/s (Rounded to nearest pstep/ vstep incrament)"""

		# print("Back velocity = ", self.back.velocity[1])

		#get positions/ angles
		self.rkp = self.rightShin.angle - self.rightThigh.angle
		self.lkp = self.leftShin.angle - self.leftThigh.angle
		self.rhp = self.rightThigh.angle - self.butt.angle
		self.lhp = self.leftThigh.angle - self.butt.angle
		self.bp = self.back.angle - self.butt.angle 
		#get height of butt off of ground
		self.buttHeight = np.floor((self.wY - self.butt.position[1] - 50)/50) #need to translate from pixel value to discrete step
		# if self.buttHeight > 4:
		# 	self.buttHeight = 4

		# if self.butt.angle < -1:
		# 	self.buttAng = -1
		# if (self.butt.angle > -1) & (self.butt.angle < 1):
		# 	self.buttAng = 0
		# if self.butt.angle > 1:
		# 	self.buttAng = 1
		
		#get vels
		self.rkv = self.rightShin.angular_velocity - self.rightThigh.angular_velocity
		self.lkv = self.leftShin.angular_velocity - self.leftThigh.angular_velocity
		self.rhv = self.rightThigh.angular_velocity - self.butt.angular_velocity
		self.lhv = self.leftThigh.angular_velocity - self.butt.angular_velocity
		self.bv = self.butt.angular_velocity - self.back.angular_velocity
		self.backv = self.back.angular_velocity

		#should probably clean this up a bit...
		statevec = np.array([self.rkp,self.lkp,self.rhp,self.lhp,self.bp,self.butt.position[1],self.butt.angle,self.rkv,self.lkv,self.rhv,self.lhv,self.bv,self.backv])
		
		self.laststate = self.statevec
		self.statevec = torch.from_numpy(statevec) #??
		# print("sv = ", self.statevec)

		return(self.statevec.float()) #cast to float (is double)
		
	def activate_joints(self, rightKneeAction, leftKneeAction, rightHipAction, leftHipAction, backAction):
		"""applys torques to joints according to input values"""

		self.rightShin.apply_force_at_local_point((-rightKneeAction*self.torqueMult,0),(0,0))
		self.rightShin.apply_force_at_local_point((rightKneeAction*self.torqueMult,0),(0,-30))		
		self.leftShin.apply_force_at_local_point((-leftKneeAction*self.torqueMult,0),(0,0))
		self.leftShin.apply_force_at_local_point((leftKneeAction*self.torqueMult,0),(0,-30))
		self.rightThigh.apply_force_at_local_point((-rightHipAction*self.torqueMult,0),(0,0))
		self.rightThigh.apply_force_at_local_point((rightHipAction*self.torqueMult,0),(0,-30))		
		self.leftThigh.apply_force_at_local_point((-leftHipAction*self.torqueMult,0),(0,0))
		self.leftThigh.apply_force_at_local_point((leftHipAction*self.torqueMult,0),(0,-30))
		self.back.apply_force_at_local_point((-backAction*self.torqueMult,0),(0,0))
		self.back.apply_force_at_local_point((backAction*self.torqueMult,0),(0,-30))
		
		# actionvec = np.array([rightHipAction,leftKneeAction,rightHipAction,leftHipAction,backAction])
		# print(actionvec)
		# self.actionvec = torch.from_numpy(actionvec)

		self.actionvec = torch.Tensor([rightHipAction,leftKneeAction,rightHipAction,leftHipAction,backAction])

	def calculate_reward(self):
		'''calculates reward for current trial'''
		# self.reward = self.back.velocity[0] + self.step*0.01 + self.back.position[0]
		# self.reward = self.back.position[0] - (self.wX/4) 
		self.reward = self.step*0.01 - self.back.position[0] + self.back.velocity[0] * 0.01

		print(self.reward)
		return(self.reward)


	def tick(self):
		"""simulates one timestep"""

		maxRunLen = 500
		if self.step > maxRunLen:
			self.fallen = True

		#upper body or butt has touched ground -> stop trial
		self.h.begin = self.fell_over

		for event in pygame.event.get():
			    if event.type == QUIT:
			        exit()

		self.calculate_reward()

		# #update history
		# self.history.add(self.laststate,self.actionvec,self.reward,self.statevec,self.done)

		# #learn if enough samples in history
		# if len(self.history) > 100: #batch_size = 100
		# 	experiences = self.history.sample()
			# self.learn(experiences, self.discountFactor)

		if self.viz:
			self.screen.fill(self.sky)
		self.space.step(1./60)
		self.space.debug_draw(self.draw_options)
		if self.viz:	
			pygame.display.flip()
			pygame.display.set_caption("fps: " + str(self.clock.get_fps()))		
		self.clock.tick(60*self.playBackSpeed)
		self.step += 1



	def run(self):
		"""runs simulation for numerous timesteps (used for debug)"""
		
		self.fallen = False
		while self.fallen == False:

			self.get_states()

			#upper body or butt has touched ground
			self.h.begin = self.fell_over
		    #timed out
			if self.step > 1000:
				self.fallen = True
				self.calculate_reward()

			for event in pygame.event.get():
			    if event.type == QUIT:
			        exit()
			    elif event.type == KEYDOWN and event.key == K_ESCAPE:
			        exit()

			    #QWOP
			    elif event.type == KEYDOWN and event.key == K_q:
			        # print("Q")
			        self.rightThigh.apply_force_at_local_point((100000,0),(0,0))
			        self.rightThigh.apply_force_at_local_point((-100000,0),(0,-30))
			    elif event.type == KEYDOWN and event.key == K_w:
			        # print("W")
			        self.leftThigh.apply_force_at_local_point((100000,0),(0,0))
			        self.leftThigh.apply_force_at_local_point((-100000,0),(0,-30))
			    elif event.type == KEYDOWN and event.key == K_o:
			        # print("O")
			        self.rightShin.apply_force_at_local_point((-100000,0),(0,0))
			        self.rightShin.apply_force_at_local_point((100000,0),(0,-30))
			    elif event.type == KEYDOWN and event.key == K_p:
			        # print("P")
			        self.leftShin.apply_force_at_local_point((-100000,0),(0,0))
			        self.leftShin.apply_force_at_local_point((100000,0),(0,-30))

			    elif event.type == KEYDOWN and event.key == K_e:
			        # print("P")
			        self.back.apply_force_at_local_point((-100000,0),(0,0))
			        self.back.apply_force_at_local_point((100000,0),(0,-30))

			# screen.fill(pygame.color.THECOLORS["yellow"])
			if self.viz:
				self.screen.fill(self.sky)

				# self.screen.blit(self.help_txt, (5, self.screen.get_height() - 20))

				mouse_pos = pygame.mouse.get_pos()

				# Display help message
				x = mouse_pos[0] / self.box_size * self.box_size
				y = mouse_pos[1] / self.box_size * self.box_size
				if (x,y) in self.box_texts:    
				    txts = self.box_texts[(x,y)]
				    i = 0
				    for txt in txts:
				        pos = (5,box_size * 2 + 10 + i*20)
				        screen.blit(txt, pos)        
				        i += 1

				self.mouse_body.position = mouse_pos

			self.space.step(1./60)

			self.space.debug_draw(self.draw_options)
			
			if self.viz:
				pygame.display.flip()
				pygame.display.set_caption("fps: " + str(self.clock.get_fps()))

			self.clock.tick(60*self.playBackSpeed)
			self.step += 1

if __name__ == "__main__":

	Epochs = 30

	for epoch in range(Epochs):
		print("Epoch # ", epoch)
		body = ragdoll(viz = True, arms = False, playBackSpeed = 10)
		# body.run()
		for i in range(100):
			body.activate_joints(np.random.rand()-0.5,np.random.rand()-0.5,np.random.rand()-0.5,np.random.rand()-0.5,np.random.rand()-0.5)
			body.tick()