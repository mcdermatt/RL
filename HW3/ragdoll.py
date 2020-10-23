import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util
import pygame
import pygame.color
from pygame.locals import *
import pickle
import pygame
import numpy as np

class ragdoll:

	torques = np.zeros([5,1])
		#right knee
		#left knee
		#right hip
		#left hip
		#back

	#how I did skrrt
	q = np.ones([3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2]) #[states, actions, (avg reward, #occurences)]

	# more memory efficient, needs model of how actions will change agent to next state
	# q = np.ones([3,3,3,3,3,3,3,3,3,3,3,2]) # [states, (avg reward, occurences)]

	def __init__(self, pol = None, q = q, viz = True, arms = True, playBackSpeed = 1):

		# self.wX = 1600
		# self.wY = 800
		self.wX = 1600
		self.wY = 600
		self.startX = self.wX / 4
		self.dampingCoeff = 10000
		self.torqueMult = 50000 #25000
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
		self.pol = pol
		self.q = q

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
		self.space.gravity = (0.0, 900.0)
		self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
		self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES


		class Box: #stolen from stack overflow
		    def __init__(ragdoll, p0=(10, 10), p1=(self.wX-10,self.wY-50), d=2):
		        x0, y0 = p0
		        x1, y1 = p1
		        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
		        for i in range(4):
		            segment = pymunk.Segment(self.space.static_body, pts[i], pts[(i+1)%4], d)
		            segment.elasticity = 1
		            segment.friction = 1
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
		self.rightShin = self.add_limb(self.space, (self.startX,500), color = self.background, friction = 1)
		self.rightThigh = self.add_limb(self.space, (self.startX,400), thiccness = 15, color = self.background)
		self.rightKnee = pymunk.PivotJoint(self.rightShin,self.rightThigh,(self.startX,450))
		self.rightKneeLimits = pymunk.RotaryLimitJoint(self.rightShin,self.rightThigh,self.kneeMin,self.kneeMax)
		self.rightKneeDamp = pymunk.DampedRotarySpring(self.rightShin,self.rightThigh,0,0,self.dampingCoeff)
		self.space.add(self.rightKnee)
		self.space.add(self.rightKneeLimits)
		self.space.add(self.rightKneeDamp)
		#add left leg
		self.leftShin = self.add_limb(self.space, (self.startX,500), color = self.midground, friction = 1)
		self.leftThigh = self.add_limb(self.space, (self.startX,400), thiccness = 15, color = self.midground)
		self.leftKnee = pymunk.PivotJoint(self.leftShin,self.leftThigh,(self.startX,450))
		self.leftKneeLimits = pymunk.RotaryLimitJoint(self.leftShin,self.leftThigh,self.kneeMin,self.kneeMax)
		self.leftKneeDamp = pymunk.DampedRotarySpring(self.leftShin,self.leftThigh,0,0,self.dampingCoeff)
		self.space.add(self.leftKneeDamp)
		self.space.add(self.leftKnee)
		self.space.add(self.leftKneeLimits)
		#add butt & hips
		buttOffset = 10 
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
		try:
			if pol == None:
				self.initPolicy()
		except:
			pass

		self.history = np.zeros([1,11])
		self.history[0,5] = 4 #sim starts with hips at pos 4, all other states 0
		self.history = self.history.astype(int)


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
	    print("player fell over at x =", self.back.position[0])
	    # pygame.quit()
	    self.fallen = True
	    self.calculate_reward()
	    self.update_values()
	    return True

	def get_states(self):
		"""gets positions and velocities of each joint in Rad and Rad/s (Rounded to nearest pstep/ vstep incrament)"""

		#get positions/ angles
		self.rkp = self.rightShin.angle - self.rightThigh.angle
		self.lkp = self.leftShin.angle - self.leftThigh.angle
		self.rhp = self.rightThigh.angle - self.butt.angle
		self.lhp = self.leftThigh.angle - self.butt.angle
		self.bp = self.back.angle - self.butt.angle 
		#get height of butt off of ground
		self.buttHeight = np.floor((self.wY - self.butt.position[1] - 50)/50) #need to translate from pixel value to discrete step
		if self.buttHeight > 4:
			self.buttHeight = 4
		#get vels
		self.rkv = self.rightShin.angular_velocity - self.rightThigh.angular_velocity
		self.lkv = self.leftShin.angular_velocity - self.leftThigh.angular_velocity
		self.rhv = self.rightThigh.angular_velocity - self.butt.angular_velocity
		self.lhv = self.leftThigh.angular_velocity - self.butt.angular_velocity
		self.bv = self.butt.angular_velocity - self.back.angular_velocity

		# print(self.rkv)
		
		statevec = np.array([self.rkp,self.lkp,self.rhp,self.lhp,self.bp,self.buttHeight,self.rkv,self.lkv,self.rhv,self.lhv,self.bv])
		# print(statevec[5])
		#round statevec to nearest incrament
		statevec[:2] = np.floor(statevec[:2]*self.pStep/(self.kneeMax-self.kneeMin))
		statevec[2:4] = np.floor(statevec[3:5]*self.pStep/(self.kneeMax-self.kneeMin)) 
		statevec[4] = np.floor(statevec[4]*self.pStep/(self.kneeMax-self.kneeMin))
		statevec[6:] = np.floor(statevec[6:]*self.vStep/np.pi) # need to make non-negative

		# print(statevec[6])

		#saturate positions outside joint limits
		for i in range(5):
			if statevec[i] < 0:
				statevec[i] = 0
			if statevec[i] > self.pStep - 1:
				statevec[i] = self.pStep - 1
		for j in range(5):
			if statevec[j+6] < 0:
				statevec[j+6] = 0
			if statevec[j+6] > self.vStep - 1:
				statevec[j+6] = self.vStep - 1

		# print(self.rkp)
		# statevec = np.floor(statevec[:2]*self.pStep/(self.kneeMax-self.kneeMin))
		
		# print(statevec)
		# return statevec
		self.statevec = statevec.astype(int)
		# self.statevec = [self.statevec]

	def activate_joints(self):
		"""applys torques to joints according to state vector and current policy"""

		# print(self.statevec)
		# print(self.pol[int(self.statevec[0]), int(self.statevec[1]), int(self.statevec[2]), int(self.statevec[3]), int(self.statevec[4]), 
		# 													int(self.statevec[5]), int(self.statevec[6]), int(self.statevec[7]), int(self.statevec[8]), int(self.statevec[9]), 
		# 													int(self.statevec[10]),0])

		#Right Knee
		self.rightShin.apply_force_at_local_point((-self.pol[int(self.statevec[0]), int(self.statevec[1]), int(self.statevec[2]), int(self.statevec[3]), int(self.statevec[4]), 
															int(self.statevec[5]), int(self.statevec[6]), int(self.statevec[7]), int(self.statevec[8]), int(self.statevec[9]), 
															int(self.statevec[10]),0]*self.torqueMult,0),(0,0))
		self.rightShin.apply_force_at_local_point((self.pol[int(self.statevec[0]), int(self.statevec[1]), int(self.statevec[2]), int(self.statevec[3]), int(self.statevec[4]), 
															int(self.statevec[5]), int(self.statevec[6]), int(self.statevec[7]), int(self.statevec[8]), int(self.statevec[9]), 
															int(self.statevec[10]),0]*self.torqueMult,0),(0,-30))
		#Left Knee
		self.leftShin.apply_force_at_local_point((-self.pol[int(self.statevec[0]), int(self.statevec[1]), int(self.statevec[2]), int(self.statevec[3]), int(self.statevec[4]), 
															int(self.statevec[5]), int(self.statevec[6]), int(self.statevec[7]), int(self.statevec[8]), int(self.statevec[9]), 
															int(self.statevec[10]),1]*self.torqueMult,0),(0,0))
		self.leftShin.apply_force_at_local_point((self.pol[int(self.statevec[0]), int(self.statevec[1]), int(self.statevec[2]), int(self.statevec[3]), int(self.statevec[4]), 
															int(self.statevec[5]), int(self.statevec[6]), int(self.statevec[7]), int(self.statevec[8]), int(self.statevec[9]), 
															int(self.statevec[10]),1]*self.torqueMult,0),(0,-30))
		#Right Thigh
		self.rightThigh.apply_force_at_local_point((-self.pol[int(self.statevec[0]), int(self.statevec[1]), int(self.statevec[2]), int(self.statevec[3]), int(self.statevec[4]), 
															int(self.statevec[5]), int(self.statevec[6]), int(self.statevec[7]), int(self.statevec[8]), int(self.statevec[9]), 
															int(self.statevec[10]),2]*self.torqueMult,0),(0,0))
		self.rightThigh.apply_force_at_local_point((self.pol[int(self.statevec[0]), int(self.statevec[1]), int(self.statevec[2]), int(self.statevec[3]), int(self.statevec[4]), 
															int(self.statevec[5]), int(self.statevec[6]), int(self.statevec[7]), int(self.statevec[8]), int(self.statevec[9]), 
															int(self.statevec[10]),2]*self.torqueMult,0),(0,-30))
		#Left Thigh
		self.leftThigh.apply_force_at_local_point((-self.pol[int(self.statevec[0]), int(self.statevec[1]), int(self.statevec[2]), int(self.statevec[3]), int(self.statevec[4]), 
															int(self.statevec[5]), int(self.statevec[6]), int(self.statevec[7]), int(self.statevec[8]), int(self.statevec[9]), 
															int(self.statevec[10]),3]*self.torqueMult,0),(0,0))
		self.leftThigh.apply_force_at_local_point((self.pol[int(self.statevec[0]), int(self.statevec[1]), int(self.statevec[2]), int(self.statevec[3]), int(self.statevec[4]), 
															int(self.statevec[5]), int(self.statevec[6]), int(self.statevec[7]), int(self.statevec[8]), int(self.statevec[9]), 
															int(self.statevec[10]),3]*self.torqueMult,0),(0,-30))
		#back
		self.back.apply_force_at_local_point((-self.pol[int(self.statevec[0]), int(self.statevec[1]), int(self.statevec[2]), int(self.statevec[3]), int(self.statevec[4]), 
															int(self.statevec[5]), int(self.statevec[6]), int(self.statevec[7]), int(self.statevec[8]), int(self.statevec[9]), 
															int(self.statevec[10]),4]*self.torqueMult,0),(0,0))
		self.back.apply_force_at_local_point((self.pol[int(self.statevec[0]), int(self.statevec[1]), int(self.statevec[2]), int(self.statevec[3]), int(self.statevec[4]), 
															int(self.statevec[5]), int(self.statevec[6]), int(self.statevec[7]), int(self.statevec[8]), int(self.statevec[9]), 
															int(self.statevec[10]),4]*self.torqueMult,0),(0,-30))
		pass

	def calculate_reward(self):
		'''calculates reward for current trial'''
		self.reward = self.back.position[0]
		pass

	def update_values(self):
		'''Updates values of each state after conclusion of trial'''
		for i in range(np.shape(self.history)[0]):
			self.pol
		pass

	def initPolicy(self):
		"""makes initial random policy"""
		print("starting new policy")
		self.pol = np.random.rand(self.pStep, self.pStep, self.pStep, self.pStep, self.pStep, 5, self.vStep, self.vStep, self.vStep, self.vStep, self.vStep, 5)
		# [rkp, lkp, rhp, lhp, bp, rkv, lkv, rhv, lhv, bv, buttHeight, joint actions]
		# Joint actions: right knee, left knee, right hip, left hip, back
		self.pol[self.pol < 0.33] = -1
		self.pol[(self.pol < 0.66) & (self.pol > 0.33)] = 0
		self.pol[self.pol > 0.66] = 1

		# print(self.pol)

	def run(self):
		step = 0
		self.fallen = False
		while self.fallen == False:

			self.get_states()
			self.activate_joints() #send torque commands to joints as func of states from current policy

			#record history of current trajectory
			self.history = np.concatenate((self.history,[self.statevec]),axis = 0)
			# print(self.history)

			#upper body or butt has touched ground
			self.h.begin = self.fell_over
		    #timed out
			if step > 1000:
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

			    #FOR DEBUG: Drag around ragdoll with mouse pointer
			    # elif event.type == MOUSEBUTTONDOWN:
			    #     if self.mouse_joint != None:
			    #         space.remove(mouse_joint)
			    #         self.mouse_joint = None

			    #     p = Vec2d(event.pos)
			    #     hit = self.space.point_query_nearest(p, 5, pymunk.ShapeFilter())
			    #     if hit != None and hit.shape.body.body_type == pymunk.Body.DYNAMIC:
			    #         shape = hit.shape
			    #         # Use the closest point on the surface if the click is outside 
			    #         # of the shape.
			    #         if hit.distance > 0:
			    #             nearest = hit.point 
			    #         else:
			    #             nearest = p
			    #         self.mouse_joint = pymunk.PivotJoint(self.mouse_body, shape.body, 
			    #             (0,0), shape.body.world_to_local(nearest))
			    #         self.mouse_joint.max_force = 50000
			    #         self.mouse_joint.error_bias = (1-0.15) ** 60
			    #         self.space.add(self.mouse_joint)
			            
			    # elif event.type == MOUSEBUTTONUP:
			    #     if self.mouse_joint != None:
			    #         self.space.remove(self.mouse_joint)
			    #         self.mouse_joint = None


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
			step += 1

if __name__ == "__main__":

	body = ragdoll(viz = True, arms = False)
	body.run()