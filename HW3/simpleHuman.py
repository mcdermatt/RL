import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util
import pygame
import pygame.color
from pygame.locals import *
import pickle
import pygame
import numpy as np

wX = 1600
wY = 600
# wX = 1000
# wY = 700

dampingCoeff = 10000

foreground = (178,102,255,255)
midground = (153,51,255,255)
background = (127,0,255,255)
sky = (32,32,32,255)
floor = (96,96,96,255)

Arms = True
# Arms = False
assist = False

#init pygame
pymunk.pygame_util.positive_y_is_up = False
pygame.init()
screen = pygame.display.set_mode((wX,wY))
clock = pygame.time.Clock()
font = pygame.font.Font(None, 24)
box_size = 200
box_texts = {}
help_txt = font.render(
    "Pymunk simple human 2D demo. Use mouse to drag/drop", 
    1, pygame.color.THECOLORS["darkgray"])
mouse_joint = None
mouse_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)

#init sim space
space = pymunk.Space()
space.gravity = (0.0, 900.0)
draw_options = pymunk.pygame_util.DrawOptions(screen)
draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES


class Box:
    def __init__(self, p0=(10, 10), p1=(wX-10,wY-50), d=2):
        x0, y0 = p0
        x1, y1 = p1
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for i in range(4):
            segment = pymunk.Segment(space.static_body, pts[i], pts[(i+1)%4], d)
            segment.elasticity = 1
            segment.friction = 1
            segment.color = sky
            space.add(segment)
Box()

def add_limb(space,pos,length = 30, mass = 2, thiccness = 10, color = (100,100,100,255), COLLTYPE = 1, elasticity = 0.1):#filter = 0b100):
	body = pymunk.Body()
	body.position = Vec2d(pos)
	shape = pymunk.Segment(body, (0,length), (0,-length), thiccness)
	shape.mass = mass
	shape.friction = 0.7
	shape.elasticity = elasticity
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
	space.add(body, shape)
	#shapes of same filter will not collide with one another
	shape.filter = pymunk.ShapeFilter(categories=filter, mask=pymunk.ShapeFilter.ALL_MASKS ^ filter)
	return body

#add right leg
rightShin = add_limb(space, (100,500), color = background, elasticity = 0.8)
# rightShin.elasticity = 0.8
rightThigh = add_limb(space, (100,400), thiccness = 15, color = background)
rightKnee = pymunk.PivotJoint(rightShin,rightThigh,(100,450))
rightKneeLimits = pymunk.RotaryLimitJoint(rightShin,rightThigh,-2.5,0)
rightKneeDamp = pymunk.DampedRotarySpring(rightShin,rightThigh,0,0,dampingCoeff)
space.add(rightKnee)
space.add(rightKneeLimits)
space.add(rightKneeDamp)

#add left leg
leftShin = add_limb(space, (100,500), color = midground, elasticity = 0.8)
# leftShin.elasticity = 0.8
leftThigh = add_limb(space, (100,400), thiccness = 15, color = midground)
leftKnee = pymunk.PivotJoint(leftShin,leftThigh,(100,450))
leftKneeLimits = pymunk.RotaryLimitJoint(leftShin,leftThigh,-2.5,0)
leftKneeDamp = pymunk.DampedRotarySpring(leftShin,leftThigh,0,0,dampingCoeff)
space.add(leftKneeDamp)
space.add(leftKnee)
space.add(leftKneeLimits)

buttOffset = 10 
butt = add_limb(space,(100-buttOffset,320), length = 10, thiccness = 17,color = midground, COLLTYPE = 3)

rightHip = pymunk.PivotJoint(rightThigh,butt,(100,350))
rightHipLimits = pymunk.RotaryLimitJoint(rightThigh,butt,-1,3)
rightHipDamp = pymunk.DampedRotarySpring(rightThigh,butt,0,0,dampingCoeff)
space.add(rightHipDamp)
space.add(rightHip)
space.add(rightHipLimits)

leftHip = pymunk.PivotJoint(leftThigh,butt,(100,350))
leftHipLimits = pymunk.RotaryLimitJoint(leftThigh,butt,-1,3)
leftHipDamp = pymunk.DampedRotarySpring(leftThigh,butt,0,0,dampingCoeff)
space.add(leftHipDamp)
space.add(leftHip)
space.add(leftHipLimits)

back = add_limb(space,(100,245), mass = 2, length = 30, thiccness = 15, color = midground, COLLTYPE = 3)# filter = 0b01)
spine = pymunk.PivotJoint(butt,back,(100,300))
spineLimits = pymunk.RotaryLimitJoint(butt,back,-0.25,1)
spineDamp = pymunk.DampedRotarySpring(butt,back,0,0,dampingCoeff)
space.add(spineDamp)
space.add(spine)
space.add(spineLimits)

if Arms:
	
	shoulderHeight = 240

	head = add_limb(space,(100,shoulderHeight - 50), length = 10, thiccness = 22,color = midground, COLLTYPE = 3)
	neck = pymunk.PivotJoint(back,head,(100, shoulderHeight - 40))
	neckLimits = pymunk.RotaryLimitJoint(back,head,-0.2,0.8)
	space.add(neck)
	space.add(neckLimits)

	#add right arm
	rightLowerArm = add_limb(space, (100,shoulderHeight+80), color = foreground, COLLTYPE = 2)
	rightUpperArm = add_limb(space, (100,shoulderHeight), thiccness = 12, color = foreground, COLLTYPE = 2)
	rightElbow = pymunk.PivotJoint(rightLowerArm,rightUpperArm,(100,shoulderHeight+50))
	rightElbowLimits = pymunk.RotaryLimitJoint(rightLowerArm,rightUpperArm,0,2.2)
	rightElbowDamp = pymunk.DampedRotarySpring(rightLowerArm,rightUpperArm,0,0,dampingCoeff)
	space.add(rightElbow)
	space.add(rightElbowLimits)
	space.add(rightElbowDamp)

	#add left arm
	shoulderHeight = 240
	leftUpperArm = add_limb(space, (100,shoulderHeight), thiccness = 12, color = background, COLLTYPE = 2)
	leftLowerArm = add_limb(space, (100,shoulderHeight+80), color = background, COLLTYPE = 2) #appearing in wierd order
				# if i comment out the lower arm the upper arm starts acting weird

	leftElbow = pymunk.PivotJoint(leftLowerArm,leftUpperArm,(100,shoulderHeight+50))
	leftElbowLimits = pymunk.RotaryLimitJoint(leftLowerArm,leftUpperArm,0,2.2)
	leftElbowDamp = pymunk.DampedRotarySpring(leftLowerArm,leftUpperArm,0,0,dampingCoeff)
	space.add(leftElbow)
	space.add(leftElbowLimits)
	space.add(leftElbowDamp)

	rightShoulder = pymunk.PivotJoint(rightUpperArm,back,(100,shoulderHeight-20))
	# rightShoulderLimits = pymunk.RotaryLimitJoint(rightUpperArm,back,-2.5,3.5)
	rightShoulderDamp = pymunk.DampedRotarySpring(rightUpperArm,back,0,0,dampingCoeff)
	space.add(rightShoulder)
	# space.add(rightShoulderLimits)
	space.add(rightShoulderDamp)

	leftShoulder = pymunk.PivotJoint(leftUpperArm,back,(100,shoulderHeight-20))
	# leftShoulderLimits = pymunk.RotaryLimitJoint(leftUpperArm,back,-2.5,3.5)
	leftShoulderDamp = pymunk.DampedRotarySpring(leftUpperArm,back,0,0,dampingCoeff)
	space.add(leftShoulder)
	# space.add(leftShoulderLimits)
	space.add(leftShoulderDamp)

	


#init collision callback function
def fell_over(space, arbiter, x):
    print("player fell over at x =", back.position[0])
    # pygame.quit()
    return True

# Create and add the "goal" 
COLLTYPE_GOAL = 2
goal_body = pymunk.Body()
goal = pymunk.Poly(goal_body, [(10,wY-50),(10,wY),(wX-10,wY),(wX-10,wY-50)])
goal.color = floor
goal.collision_type = COLLTYPE_GOAL
space.add(goal)

COLLTYPE_BACK = 3
h = space.add_collision_handler(COLLTYPE_BACK, COLLTYPE_GOAL)

while True:

	print(back.angle)
	
	# print(back.angle)

	leftKneeAng = leftShin.angle - leftThigh.angle
	rightKneeAng = rightShin.angle - rightThigh.angle
	# print(leftKneeAng,rightKneeAng)
	h.begin = fell_over
    
	# back.apply_force_at_world_point((0,1000),(0,0))
	if assist == True:
		back.apply_force_at_local_point((-10000*np.sin(back.angle),0),(0,0))

	for event in pygame.event.get():
	    if event.type == QUIT:
	        exit()
	    elif event.type == KEYDOWN and event.key == K_ESCAPE:
	        exit()

	    #QWOP
	    elif event.type == KEYDOWN and event.key == K_q:
	        # print("Q")
	        rightThigh.apply_force_at_local_point((100000,0),(0,0))
	        rightThigh.apply_force_at_local_point((-100000,0),(0,-30))
	    elif event.type == KEYDOWN and event.key == K_w:
	        # print("W")
	        leftThigh.apply_force_at_local_point((100000,0),(0,0))
	        leftThigh.apply_force_at_local_point((-100000,0),(0,-30))
	    elif event.type == KEYDOWN and event.key == K_o:
	        # print("O")
	        rightShin.apply_force_at_local_point((-100000,0),(0,0))
	        rightShin.apply_force_at_local_point((100000,0),(0,-30))
	    elif event.type == KEYDOWN and event.key == K_p:
	        # print("P")
	        leftShin.apply_force_at_local_point((-100000,0),(0,0))
	        leftShin.apply_force_at_local_point((100000,0),(0,-30))
	    elif event.type == KEYDOWN and event.key == K_e:
			        # print("P")
			        back.apply_force_at_local_point((-100000,0),(0,0))
			        back.apply_force_at_local_point((100000,0),(0,-30))

	    elif event.type == MOUSEBUTTONDOWN:
	        if mouse_joint != None:
	            space.remove(mouse_joint)
	            mouse_joint = None

	        p = Vec2d(event.pos)
	        hit = space.point_query_nearest(p, 5, pymunk.ShapeFilter())
	        if hit != None and hit.shape.body.body_type == pymunk.Body.DYNAMIC:
	            shape = hit.shape
	            # Use the closest point on the surface if the click is outside 
	            # of the shape.
	            if hit.distance > 0:
	                nearest = hit.point 
	            else:
	                nearest = p
	            mouse_joint = pymunk.PivotJoint(mouse_body, shape.body, 
	                (0,0), shape.body.world_to_local(nearest))
	            mouse_joint.max_force = 50000
	            mouse_joint.error_bias = (1-0.15) ** 60
	            space.add(mouse_joint)
	            
	    elif event.type == MOUSEBUTTONUP:
	        if mouse_joint != None:
	            space.remove(mouse_joint)
	            mouse_joint = None

	#check to see if body of player has collided with box


	# screen.fill(pygame.color.THECOLORS["yellow"])
	screen.fill(sky)

	screen.blit(help_txt, (5, screen.get_height() - 20))

	mouse_pos = pygame.mouse.get_pos()

	# Display help message
	x = mouse_pos[0] / box_size * box_size
	y = mouse_pos[1] / box_size * box_size
	if (x,y) in box_texts:    
	    txts = box_texts[(x,y)]
	    i = 0
	    for txt in txts:
	        pos = (5,box_size * 2 + 10 + i*20)
	        screen.blit(txt, pos)        
	        i += 1

	mouse_body.position = mouse_pos

	space.step(1./60)

	space.debug_draw(draw_options)
	pygame.display.flip()

	clock.tick(60)
	pygame.display.set_caption("fps: " + str(clock.get_fps()))