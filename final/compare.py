from viz import viz
from statePredictor import statePredictor
import numpy as np 


class compare:

	x0 = np.array([0,0.5,1.5,0,0,0])
	rf = np.array([1,1,1,0.5,0.5,0.5,0.025,0.025,0.025])
	ef = np.array([1,1,1,0.5,0.5,0.5,0.025,0.025,0.025])
	externalLoads = np.zeros(3)

	def __init__(self, realFric = rf, estFric = ef, externalLoads = externalLoads, x0 = x0, gravity = True, render = True):

		""" realFric - ground truth real friction of simulated arm
			estFric  - estimated friction params of each joint
					
					[static0,  static1, static2,
					 kinetic0, kinetic, kinetic2,
					 damp0,	   damp1,   damp2    ]

			externalLoads - forces on end effector in xyz space
			gravity - will gravity be compensated by seperate controller """

		self.realFric = realFric
		self.estFric = estFric
		self.externalLoads = externalLoads
		self.gravity = gravity
		self.isViz = viz
		self.x0 = x0

		self.sp = statePredictor()
		self.sp.dt = 1/60
		self.sp.numPts = 120
		self.sp.x0 = self.x0

		#init ground truth arms params
		self.sp.numerical_constants[12:] = self.realFric
		self.path1 = self.sp.predict()
		print(self.path1)

		#init estimated arms params
		self.sp.numerical_constants[12:] = self.estFric
		self.path2 = self.sp.predict()
		print(self.path2)

	

	def render(self):
		#shows estimated friction path for now
		v = viz(self.path2, self.path1, use_GPU = False)
		v.start()


if __name__ == "__main__":

	ef = np.array([.5,.5,.5,0.75,0.75,0.75,0.125,0.125,0.125])

	c = compare(estFric = ef)
	c.render()