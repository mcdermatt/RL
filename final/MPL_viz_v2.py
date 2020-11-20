import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from statePredictor import statePredictor
from time import sleep

#calculates or plays back pre-recorded trajectory (runs much smoother than viz1)

presolved = True
dt = 0.02

sp = statePredictor()
fig = plt.figure()
ax = fig.add_subplot(111, xlim=(-1,1), ylim=(-1,1), zlim=(-1,1), projection='3d', autoscale_on=False)
ax.grid(False)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
plt.xlabel("x",fontdict=None,labelpad=None)
plt.ylabel("y",fontdict=None,labelpad=None)
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')	
base, = plt.plot([0],[0],[0],'bo', markersize = 8)

x0 = np.array([0,np.deg2rad(30),np.deg2rad(90),0,0,0])

if presolved is False:
	runLen = 150
	#generate solution vector
	y = np.zeros([runLen,3])
	for t in range(runLen):

		nextStates = sp.predict(x0 = x0, dt = dt)[1]
		x0 = nextStates
		y[t] = nextStates[:3]
		# print(nextStates[:3])
		print("step ", t, " of ", runLen)
	np.save("path1", y)
else:
	y = np.load("path1.npy")
	runLen = len(y)

while True:
	for i in range(runLen):
		#debug, uncomment when done
		# get elbow position
		xElb =  0.5 * np.sin(y[i,0])*np.sin(y[i,1])
		zElb =  0.5 * np.cos((y[i,1])) 
		yElb =   0.5 * np.cos(y[i,0])*np.sin(y[i,1])

		# link2RotEff = link1Rot + link2Rot
		xHan = xElb +  0.5 * np.sin(y[i,0])*np.sin((y[i,1] + y[i,2]))
		zHan = zElb +  0.5 * np.cos(((y[i,1] + y[i,2]))) 
		yHan = yElb +  0.5 * np.cos(y[i,0])*np.sin((y[i,1] + y[i,2]))

		try:
			elbow.remove()
			hand.remove()
			armline.remove()
		except:
			pass
		elbow, = plt.plot([xElb],[yElb],[zElb],'bo', markersize = 8)

		armline, = plt.plot([0,xElb,xHan],[0,yElb,yHan],[0,zElb,zHan], 'b-', lw = 6)

		hand, = plt.plot([xHan],[yHan],[zHan],'bo', markersize = 8)


		plt.draw()
		plt.pause(dt*2)
		# plt.cla()

plt.pause(5)