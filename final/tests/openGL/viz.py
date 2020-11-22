import ctypes
from pyglet.gl import *
from pyglet.window import key
from pywavefront import visualization, Wavefront
import numpy
import time

class viz:

	def __init__(self, path, use_GPU = False):

		self.path = path

		if use_GPU is False:
			window = pyglet.window.Window(width=1280,height=720)
		else:
			config = pyglet.gl.Config(sample_buffers=1, samples=9) #samples = number of points used for AA
			window = pyglet.window.Window(width=1280, height=720, config = config)

		keys = key.KeyStateHandler()
		window.push_handlers(keys)

		base = Wavefront('base.obj')
		link0 = Wavefront('l0.obj')
		link1 = Wavefront('l1.obj')
		link2 = Wavefront('l2andl3Dummy.obj')
		greenCheck = pyglet.image.load('greenCheck.png')
		gc = pyglet.sprite.Sprite(img=greenCheck)
		gc.scale = 0.01
		gc.x = -10
		gc.y = 12
		redX = pyglet.image.load('redX.png')
		rx = pyglet.sprite.Sprite(img=redX)
		rx.scale = 0.005
		rx.x = -10
		rx.y = 12


if __name__ == "__main__":

	path = "C:/Users/Matt/comp138/final/path1.npy"
	pathArr = numpy.load(path)
	viz = viz(pathArr, use_GPU=True)