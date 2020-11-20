import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from statePredictor import statePredictor
from time import sleep

#calcualtes and plays back trajectory

sp = statePredictor()
fig = plt.figure()
ax = fig.add_subplot(111, xlim=(-1,1), ylim=(-1,1), zlim=(-1,1), projection='3d', autoscale_on=False)
ax.grid(False)
plt.xlabel("x",fontdict=None,labelpad=None)
plt.ylabel("y",fontdict=None,labelpad=None)
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')

base, = plt.plot([0],[0],[0],'bo')


runLen = 30
numSteps = 11
steps = np.linspace(0,runLen,numSteps)
x0 = np.array([0,np.deg2rad(30),np.deg2rad(90),0,0,0])

for _ in range(runLen):
# for t in steps:

	# sp.dt = 0.1
	# sp.dt = t
	nextStates = sp.predict(x0 = x0, dt = 0.05)[1]
	x0 = nextStates
	# print(nextStates[:3])


	#debug, uncomment when done
	#get elbow position
	xElb =  0.5 * np.sin(nextStates[0])*np.sin(nextStates[1])
	zElb =  0.5 * np.cos((nextStates[1])) 
	yElb =   0.5 * np.cos(nextStates[0])*np.sin(nextStates[1])

	# link2RotEff = link1Rot + link2Rot
	xHan = xElb +  0.5 * np.sin(nextStates[0])*np.sin((nextStates[1] + nextStates[2]))
	zHan = zElb +  0.5 * np.cos(((nextStates[1] + nextStates[2]))) 
	yHan = yElb +  0.5 * np.cos(nextStates[0])*np.sin((nextStates[1] + nextStates[2]))

	try:
		elbow.remove()
		hand.remove()
		armline.remove()
	except:
		pass
	elbow, = plt.plot([xElb],[yElb],[zElb],'bo')

	armline, = plt.plot([0,xElb,xHan],[0,yElb,yHan],[0,zElb,zHan], 'b-', lw = 4)

	hand, = plt.plot([xHan],[yHan],[zHan],'bo')


	plt.draw()
	plt.pause(0.01)
	# plt.cla()

plt.pause(5)