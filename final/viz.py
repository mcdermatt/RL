import ctypes
from pyglet.gl import *
from pyglet.window import key
from pywavefront import visualization, Wavefront
import numpy as np
import time
from pyglet.window import mouse
from trajPlotter import trajPlotter

class viz:
	"""Robotic arm visualization class made using OpenGL
		(path( 3*n numpy array), use_GPU = False)
	 .start() to run"""

	def __init__(self, path,path2, use_GPU = False):

		self.path = path
		self.path2 = path2
		self.lenPath = len(path)
		self.tp = trajPlotter(self.path,self.path2)

		if use_GPU is False:
			self.window = pyglet.window.Window(width=1280,height=720)
		else:
			config = pyglet.gl.Config(sample_buffers=1, samples=9) #samples = number of points used for AA
			self.window = pyglet.window.Window(width=1280, height=720, config = config)

		self.keys = key.KeyStateHandler()
		self.window.push_handlers(self.keys)
		self.window.push_handlers(self.on_mouse_drag)

		self.base = Wavefront('assets/base.obj')
		self.link0 = Wavefront('assets/l0.obj')
		self.link1 = Wavefront('assets/l1.obj')
		self.link2 = Wavefront('assets/l2andl3Dummy.obj')
		greenCheck = pyglet.image.load('assets/greenCheck.png')
		self.gc = pyglet.sprite.Sprite(img=greenCheck)
		self.gc.scale = 0.01
		self.gc.x = -10
		self.gc.y = 12
		redX = pyglet.image.load('assets/redX.png')
		self.rx = pyglet.sprite.Sprite(img=redX)
		self.rx.scale = 0.005
		self.rx.x = -10
		self.rx.y = 12

		self.l1 = 6.5
		self.l2 = 6.5
		self.l3 = 2.65
		self.rotation = 0
		self.cameraZ = 0.0
		self.i = 0
		self.on_resize(1280,720)

		self.dx = 0
		self.dy = 0
		self.theta = 0
		self.phi = 0

		#test
		#TODO generate plot from here
		self.label = None
		plotFigPath = "pathFig.png"
		plotFig = pyglet.image.load(plotFigPath)


		self.plotFig = pyglet.sprite.Sprite(img=plotFig)
		self.plotFig.scale = 0.0375
		self.plotFig.x = -32
		self.plotFig.y = 5

		self.spf = 1/60 #seconds per frame

	def on_resize(self,width, height):
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective(50., float(width)/height, 1., 100.) #change first argument for fov
		glTranslatef(0,-5,-40) #fits arm into camera view
		glMatrixMode(GL_MODELVIEW)
		return True

	def on_draw(self):
		self.window.clear()
		glClearColor(0.25,0.25,0.25,0.5) #sets background color
		glViewport(0,0,1280,720)
		glLoadIdentity()
		glMatrixMode(GL_PROJECTION)
		# glRotatef(0,0,1,0)
		

		#new
		self.plotFig.draw() #draws sprite directly to screen regardless of camera angle
		if self.label:
			self.label.draw()
		glTranslatef(self.dx/20,self.dy/20,0)
		glRotatef(self.theta/5,0,1,0)
		#not sure how important this one is
		# glRotatef(self.phi/5,-1,0,0)

		



		glMatrixMode(GL_MODELVIEW)
		link0Rot = (180/np.pi)*self.path[self.i,0]
		link1Rot = (180/np.pi)*self.path[self.i,1]
		link2Rot = (180/np.pi)*self.path[self.i,2]

		lightfv = ctypes.c_float * 4

		link2RotEff = link1Rot + link2Rot

		xElb = ( self.l1 * np.sin(link0Rot*(np.pi/180))*np.sin(link1Rot*(np.pi/180)))
		yElb = ( self.l1 * np.cos((link1Rot*(np.pi/180)))) 
		zElb =  ( self.l1 * np.cos(link0Rot*(np.pi/180))*np.sin(link1Rot*(np.pi/180)))

		xl3 = xElb + ( self.l2 * np.sin(link0Rot*(np.pi/180))*np.sin(link2RotEff*(np.pi/180)))
		yl3 = yElb + ( self.l2 * np.cos((link2RotEff*(np.pi/180)))) 
		zl3 = zElb + ( self.l2 * np.cos(link0Rot*(np.pi/180))*np.sin(link2RotEff*(np.pi/180)))

		xl4 = xElb + ( (self.l2+self.l3) * np.sin(link0Rot*(np.pi/180))*np.sin(link2RotEff*(np.pi/180)))
		yl4 = yElb + ( (self.l2+self.l3) * np.cos((link2RotEff*(np.pi/180)))) 
		zl4 = zElb + ( (self.l2+self.l3) * np.cos(link0Rot*(np.pi/180))*np.sin(link2RotEff*(np.pi/180)))

		#glLightfv(GL_LIGHT0, GL_POSITION, lightfv(-1.0, 1.0*np.sin(rotation*0.1), 1.0, 0.0))
		# glLightfv(GL_LIGHT0, GL_AMBIENT, lightfv(0.5,0.5,0.5,0.1))
		glLightfv(GL_LIGHT0, GL_DIFFUSE, lightfv(0.5, 0.5, 0.5, 0.9))
		glLightfv(GL_LIGHT0, GL_SPECULAR, lightfv(0.0,0.0,0.0,0.1))

		glEnable(GL_DEPTH_TEST)

		self.draw_base(self.base)
		self.draw_link0(self.link0, 0, 0, 0, link0Rot)
		self.draw_link1(self.link1, 0, 0, 0,link0Rot, link1Rot)
		self.draw_link2(self.link2, xElb, yElb, zElb, link0Rot, link1Rot, link2Rot)


		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
		
		## draw workspace boundary
		# self.draw_cube()
		# #draw green check if EE is inside the workspace
		# if (xl3 > -8) and (xl3 < 8) and (yl3 < 10) and (yl3 > -5) and (zl3 > 3) and (zl3 < 13):
		# 		self.gc.draw()
		# #if EE is outside workspace draw the red x
		# else:
		# 		self.rx.draw()

		# self.window.flip()
		time.sleep(0.01)

	def draw_base(self,link):
		glLoadIdentity()
		glMatrixMode(GL_MODELVIEW)
		glRotatef(45,0,1,0)
		glTranslatef(0,-3.4,0)
		visualization.draw(link)

	def draw_cube(self):
		glLoadIdentity()
		# glDisable(GL_POLYGON_SMOOTH)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
		glLineStipple(10, 0xAAAA)
		glEnable(GL_LINE_STIPPLE)
		glColor3f(0.8, 0.0, 0.1)
		glLineWidth(1)
		#robot right
		glBegin(GL_QUADS)
		glVertex3f( -8, -5,  13 )
		glVertex3f( -8,  10,  13 )
		glVertex3f( -8,  10,  3 )
		glVertex3f( -8, -5,  3 )
		glEnd()
		#robot left
		glBegin(GL_QUADS)
		glVertex3f( 8, -5,  13 )
		glVertex3f( 8,  10,  13 )
		glVertex3f( 8,  10,  3 )
		glVertex3f( 8, -5,  3 )
		glEnd()
		#robot top
		glBegin(GL_QUADS)
		glVertex3f( -8,  10,  13 )
		glVertex3f( -8,  10,  3 )
		glVertex3f( 8,  10,  3 )
		glVertex3f( 8,  10,  13 )
		glEnd()
		#robot bottom
		glBegin(GL_QUADS)
		glVertex3f( -8,  -5,  13 )
		glVertex3f( -8,  -5,  3 )
		glVertex3f( 8,  -5,  3 )
		glVertex3f( 8,  -5,  13 )
		glEnd()

		#returns polygon mode to smooth
		# glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
		glPolygonMode(GL_FRONT, GL_FILL)
		glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
		# glEnable(GL_BLEND)
		glEnable(GL_MULTISAMPLE)
		# glfwWindowHint(GLFW_SAMPLES, 4)
		glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
		# glDisable(GL_DEPTH_TEST)

	def draw_link0(self,link, x, y, z, link0Rot):
		glLoadIdentity()
		glMatrixMode(GL_MODELVIEW)
		glRotatef(link0Rot, 0.0, 1.0 , 0.0)
		glTranslatef(x, y, z)

		visualization.draw(link)

	def draw_link1(self,link, x, y, z,link0Rot, link1Rot):
		glLoadIdentity()
		glMatrixMode(GL_MODELVIEW)
		#glRotatef(180,0,1,0) #flips l1 around so it isnt bending backwards
		glRotatef(link0Rot, 0.0, 1.0 , 0.0)
		glRotatef(link1Rot, 1.0, 0.0 , 0.0)
		visualization.draw(link)
		
	def draw_link2(self,link, x, y, z, link0Rot, link1Rot, link2Rot):
		glLoadIdentity()
		glMatrixMode(GL_MODELVIEW)
		glTranslatef(x, y, z)
		#print("link0Rot: ", link0Rot, " Link1Rot: ", link1Rot, " Link2Rot: ", link2Rot)
		glRotatef(link0Rot, 0.0, 1.0 , 0.0)
		glRotatef(link1Rot, 1.0, 0.0 , 0.0)
		glRotatef(link2Rot, 1.0, 0.0, 0.0)

		visualization.draw(link)

	# @window.event #doesen't work in class
	def on_mouse_drag(self,x,y,dx,dy,button,modifiers):
		"""move camera with mouse (click and drag)"""
		
		if button == pyglet.window.mouse.LEFT:
			self.dx += dx
			self.dy += dy

		if button == pyglet.window.mouse.RIGHT:
			self.theta += dx
			self.phi += dy


	def update(self, dt):
		self.on_draw()
		self.on_resize(1280,720)

		#update text inside label here
		self.label = pyglet.text.Label('Hello, world',
                          font_name='Times New Roman',
                          font_size=12,
                          x=self.window.width//2, y=self.window.height//2,
                          anchor_x='center', anchor_y='center',color = (255,255,255,100))

		self.i += 1
		if self.i == (self.lenPath - 1):
			self.i = 0 #loop

	def start(self):
		# pyglet.clock.schedule(self.update)
		pyglet.clock.schedule_interval(self.update, self.spf)
		pyglet.app.run()


if __name__ == "__main__":

	filename1 = "path_y1.npy"
	filename2 = "path_y2.npy"
	path1 = np.load(filename1)
	path2 = np.load(filename2)
	viz = viz(path1, path2, use_GPU=False)

	viz.start()
