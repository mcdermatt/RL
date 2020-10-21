import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util
import pygame
import pygame.color
from pygame.locals import *
import pickle
import pygame

#TODO:
# take input from policy
# run()
# viz on/off
# fast forward(?)

class ragdoll:

	def __init__(self,viz = True, arms = True):

		self.wX = 1600
		self.wY = 800
		self.dampingCoeff = 10000
		self.foreground = (178,102,255,255) #foreground color
		self.midground = (153,51,255,255) 
		self.background = (127,0,255,255)
		self.sky = (153,204,255,255)
		self.floor = (0,51,102,255)

		self.screen = pygame.display.set_mode((self.wX,self.wY))
		self.clock = pygame.time.Clock()
		self.viz = viz

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
		#add right leg
		self.rightShin = self.add_limb(self.space, (100,500), color = self.background)
		self.rightThigh = self.add_limb(self.space, (100,400), thiccness = 15, color = self.background)
		self.rightKnee = pymunk.PivotJoint(self.rightShin,self.rightThigh,(100,450))
		self.rightKneeLimits = pymunk.RotaryLimitJoint(self.rightShin,self.rightThigh,-2.5,0)
		self.rightKneeDamp = pymunk.DampedRotarySpring(self.rightShin,self.rightThigh,0,0,self.dampingCoeff)
		self.space.add(self.rightKnee)
		self.space.add(self.rightKneeLimits)
		self.space.add(self.rightKneeDamp)
		#add left leg
		self.leftShin = self.add_limb(self.space, (100,500), color = self.midground)
		self.leftThigh = self.add_limb(self.space, (100,400), thiccness = 15, color = self.midground)
		self.leftKnee = pymunk.PivotJoint(self.leftShin,self.leftThigh,(100,450))
		self.leftKneeLimits = pymunk.RotaryLimitJoint(self.leftShin,self.leftThigh,-2.5,0)
		self.leftKneeDamp = pymunk.DampedRotarySpring(self.leftShin,self.leftThigh,0,0,self.dampingCoeff)
		self.space.add(self.leftKneeDamp)
		self.space.add(self.leftKnee)
		self.space.add(self.leftKneeLimits)
		#add butt & hips
		buttOffset = 10 
		self.butt = self.add_limb(self.space,(100-buttOffset,320), length = 10, thiccness = 17,color = self.midground, COLLTYPE = 3)
		self.rightHip = pymunk.PivotJoint(self.rightThigh,self.butt,(100,350))
		self.rightHipLimits = pymunk.RotaryLimitJoint(self.rightThigh,self.butt,-1,3)
		self.rightHipDamp = pymunk.DampedRotarySpring(self.rightThigh,self.butt,0,0,self.dampingCoeff)
		self.space.add(self.rightHipDamp)
		self.space.add(self.rightHip)
		self.space.add(self.rightHipLimits)
		self.leftHip = pymunk.PivotJoint(self.leftThigh,self.butt,(100,350))
		self.leftHipLimits = pymunk.RotaryLimitJoint(self.leftThigh,self.butt,-1,3)
		self.leftHipDamp = pymunk.DampedRotarySpring(self.leftThigh,self.butt,0,0,self.dampingCoeff)
		self.space.add(self.leftHipDamp)
		self.space.add(self.leftHip)
		self.space.add(self.leftHipLimits)
		#add back
		self.back = self.add_limb(self.space,(100,245), mass = 2, length = 30, thiccness = 15, color = self.midground, COLLTYPE = 3)# filter = 0b01)
		self.spine = pymunk.PivotJoint(self.butt,self.back,(100,300))
		self.spineLimits = pymunk.RotaryLimitJoint(self.butt,self.back,-0.25,1)
		self.spineDamp = pymunk.DampedRotarySpring(self.butt,self.back,0,0,self.dampingCoeff)
		self.space.add(self.spineDamp)
		self.space.add(self.spine)
		self.space.add(self.spineLimits)
		if arms:
			#add head
			shoulderHeight = 240
			self.head = self.add_limb(self.space,(100,shoulderHeight - 50), length = 10, thiccness = 22,color = self.midground, COLLTYPE = 3)
			self.neck = pymunk.PivotJoint(self.back,self.head,(100, shoulderHeight - 40))
			self.neckLimits = pymunk.RotaryLimitJoint(self.back,self.head,-0.2,0.8)
			self.space.add(self.neck)
			self.space.add(self.neckLimits)
			#add right arm
			self.rightLowerArm = self.add_limb(self.space, (100,shoulderHeight+80), color = self.foreground, COLLTYPE = 2)
			self.rightUpperArm = self.add_limb(self.space, (100,shoulderHeight), thiccness = 12, color = self.foreground, COLLTYPE = 2)
			self.rightElbow = pymunk.PivotJoint(self.rightLowerArm,self.rightUpperArm,(100,shoulderHeight+50))
			self.rightElbowLimits = pymunk.RotaryLimitJoint(self.rightLowerArm,self.rightUpperArm,0,2.2)
			self.rightElbowDamp = pymunk.DampedRotarySpring(self.rightLowerArm,self.rightUpperArm,0,0,self.dampingCoeff)
			self.space.add(self.rightElbow)
			self.space.add(self.rightElbowLimits)
			self.space.add(self.rightElbowDamp)
			#add left arm
			self.leftUpperArm = self.add_limb(self.space, (100,shoulderHeight), thiccness = 12, color = self.background, COLLTYPE = 2)
			self.leftLowerArm = self.add_limb(self.space, (100,shoulderHeight+80), color = self.background, COLLTYPE = 2)
			self.leftElbow = pymunk.PivotJoint(self.leftLowerArm,self.leftUpperArm,(100,shoulderHeight+50))
			self.leftElbowLimits = pymunk.RotaryLimitJoint(self.leftLowerArm,self.leftUpperArm,0,2.2)
			self.leftElbowDamp = pymunk.DampedRotarySpring(self.leftLowerArm,self.leftUpperArm,0,0,self.dampingCoeff)
			self.space.add(self.leftElbow)
			self.space.add(self.leftElbowLimits)
			self.space.add(self.leftElbowDamp)
			self.rightShoulder = pymunk.PivotJoint(self.rightUpperArm,self.back,(100,shoulderHeight-20))
			self.rightShoulderDamp = pymunk.DampedRotarySpring(self.rightUpperArm,self.back,0,0,self.dampingCoeff)
			self.space.add(self.rightShoulder)
			self.space.add(self.rightShoulderDamp)
			self.leftShoulder = pymunk.PivotJoint(self.leftUpperArm,self.back,(100,shoulderHeight-20))
			self.leftShoulderDamp = pymunk.DampedRotarySpring(self.leftUpperArm,self.back,0,0,self.dampingCoeff)
			self.space.add(self.leftShoulder)
			self.space.add(self.leftShoulderDamp)

		# Create and add the "goal" 
		COLLTYPE_GOAL = 2
		goal_body = pymunk.Body()
		goal = pymunk.Poly(goal_body, [(10,self.wY-50),(10,self.wY),(self.wX-10,self.wY),(self.wX-10,self.wY-50)])
		goal.color = self.floor
		goal.collision_type = COLLTYPE_GOAL
		self.space.add(goal)

		COLLTYPE_BACK = 3
		self.h = self.space.add_collision_handler(COLLTYPE_BACK, COLLTYPE_GOAL)

	def add_limb(self,space,pos,length = 30, mass = 2, thiccness = 10, color = (100,100,100,255), COLLTYPE = 1):#filter = 0b100):
		body = pymunk.Body()
		body.position = Vec2d(pos)
		shape = pymunk.Segment(body, (0,length), (0,-length), thiccness)
		shape.mass = mass
		shape.friction = 0.7
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
	    return True

	def run(self):
		while True:

			leftKneeAng = self.leftShin.angle - self.leftThigh.angle
			rightKneeAng = self.rightShin.angle - self.rightThigh.angle
			# print(leftKneeAng,rightKneeAng)
			self.h.begin = self.fell_over
		    
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
			    # elif event.type == KEYDOWN and event.key == K_e:
			    #     print("E")
			    #     back.apply_impulse_at_local_point((1000,0),(0,0))

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

			#check to see if body of player has collided with box


			# screen.fill(pygame.color.THECOLORS["yellow"])
			if self.viz:
				self.screen.fill(self.sky)

				self.screen.blit(self.help_txt, (5, self.screen.get_height() - 20))

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

			self.clock.tick(60)


if __name__ == "__main__":

	body = ragdoll(viz = True, arms = False)
	body.run()