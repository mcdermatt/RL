import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from statePredictor import statePredictor
from time import sleep
#TODO- learn clock scheduling, draw every n timesteps

sp = statePredictor()
fig = plt.figure()
ax = fig.add_subplot(111, xlim=(-1,1), ylim=(-1,1), zlim=(0,1), projection='3d', autoscale_on=False)
# ax.grid(False)
plt.xlabel("x",fontdict=None,labelpad=None)
plt.ylabel("y",fontdict=None,labelpad=None)
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')

base, = plt.plot([0],[0],[0],'bo')


runLen = 30
numSteps = 11
steps = np.linspace(0,runLen,numSteps)
sp.x0 = np.array([0,30,90,0,0,0])

for _ in range(runLen):

	sp.dt = 0.1
	nextStates = sp.predict()[1]
	sp.x0 = nextStates
	print(nextStates[:3])
	#get elbow position
	xElb = ( 0.5 * np.sin(nextStates[0]*(np.pi/180))*np.sin(nextStates[1]*(np.pi/180)))
	yElb = ( 0.5 * np.cos((nextStates[1]*(np.pi/180)))) 
	zElb =  ( 0.5 * np.cos(nextStates[0]*(np.pi/180))*np.sin(nextStates[1]*(np.pi/180)))

	# link2RotEff = link1Rot + link2Rot
	xHan = xElb + ( 0.5 * np.sin(nextStates[0]*(np.pi/180))*np.sin((nextStates[1] + nextStates[2])*(np.pi/180)))
	yHan = yElb + ( 0.5 * np.cos(((nextStates[1] + nextStates[2])*(np.pi/180)))) 
	zHan = zElb + ( 0.5 * np.cos(nextStates[0]*(np.pi/180))*np.sin((nextStates[1] + nextStates[2])*(np.pi/180)))

	try:
		elbow.remove()
		hand.remove()
	except:
		pass
	elbow, = plt.plot([xElb],[zElb],[yElb],'bo')
	hand, = plt.plot([xHan],[yHan],[zHan],'ro')

	plt.draw()
	plt.pause(0.01)
	# plt.cla()

plt.pause(5)