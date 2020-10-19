import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util
import pygame
import pygame.color
from pygame.locals import *
import pickle
import pygame

pymunk.pygame_util.positive_y_is_up = False

#init pygame
pygame.init()
screen = pygame.display.set_mode((2050, 800))
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

class Box:
    def __init__(self, p0=(10, 10), p1=(2000, 700), d=2):
        x0, y0 = p0
        x1, y1 = p1
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for i in range(4):
            segment = pymunk.Segment(space.static_body, pts[i], pts[(i+1)%4], d)
            segment.elasticity = 1
            segment.friction = 1
            space.add(segment)

def add_limb(space,pos,length = 30, mass = 2, thiccness = 10, color = (100,100,100,255), filter = 0b100):
	body = pymunk.Body()
	body.position = Vec2d(pos)
	shape = pymunk.Segment(body, (0,length), (0,-length), thiccness)
	shape.mass = mass
	shape.friction = 0.7
	shape.color = color
	COLLTYPE_BACK = 3
	shape.collision_type = COLLTYPE_BACK
	space.add(body, shape)
	#shapes of same filter will not collide with one another
	shape.filter = pymunk.ShapeFilter(categories=filter, mask=pymunk.ShapeFilter.ALL_MASKS ^ filter)
	return body

Box()

#add right leg
rightShin = add_limb(space, (100,500), color = (76,0,151,255))
rightThigh = add_limb(space, (100,400), thiccness = 15, color = (76,0,151,255))
rightKnee = pymunk.PivotJoint(rightShin,rightThigh,(100,450))
rightKneeLimits = pymunk.RotaryLimitJoint(rightShin,rightThigh,-2.5,0)
space.add(rightKnee)
space.add(rightKneeLimits)
#add left leg
leftShin = add_limb(space, (100,500), color = (178,102,255,255))
leftThigh = add_limb(space, (100,400), thiccness = 15, color = (178,102,255,255))
leftKnee = pymunk.PivotJoint(leftShin,leftThigh,(100,450))
leftKneeLimits = pymunk.RotaryLimitJoint(leftShin,leftThigh,-2.5,0)
space.add(leftKnee)
space.add(leftKneeLimits)

butt = add_limb(space,(90,320), length = 10, thiccness = 17,color = (153,51,255,255),filter = 0b01)
rightHip = pymunk.PivotJoint(rightThigh,butt,(100,350))
rightHipLimits = pymunk.RotaryLimitJoint(rightThigh,butt,-1,3)
space.add(rightHip)
space.add(rightHipLimits)
leftHip = pymunk.PivotJoint(leftThigh,butt,(100,350))
leftHipLimits = pymunk.RotaryLimitJoint(leftThigh,butt,-1,3)
space.add(leftHip)
space.add(leftHipLimits)

back = add_limb(space,(100,245), mass = 2, length = 30, thiccness = 15, color = (153,51,255,255), filter = 0b01)
spine = pymunk.PivotJoint(butt,back,(100,300))
spineLimits = pymunk.RotaryLimitJoint(butt,back,-0.25,1)
space.add(spine)
space.add(spineLimits)

#init collision callback function
def fell_over(space, arbiter, x):
    print("player fell over")
    return True

# Create and add the "goal" 
COLLTYPE_GOAL = 2
goal_body = pymunk.Body()
goal_body.position = 100,100
goal = pymunk.Circle(goal_body, 50)
goal.collision_type = COLLTYPE_GOAL
space.add(goal)

COLLTYPE_BACK = 3
back.collision_type = COLLTYPE_BACK
# rightThigh.collision_type = COLLTYPE_GROUND
h = space.add_collision_handler(COLLTYPE_BACK, COLLTYPE_GOAL)

while True:
    h.begin = fell_over

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
        # elif event.type == KEYDOWN and event.key == K_e:
        #     print("E")
        #     back.apply_impulse_at_local_point((1000,0),(0,0))

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


    screen.fill(pygame.color.THECOLORS["white"])

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