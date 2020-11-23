import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from statePredictor import statePredictor
from time import sleep

#compares trajectory of two arms with different parameters

presolved = False
inputForces = False
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
base1, = plt.plot([-0.5],[0],[0],'bo', markersize = 8)
base2, = plt.plot([0.5],[0],[0],'go', markersize = 8)

# x01 = np.array([0,np.deg2rad(30),np.deg2rad(90),0,0,0])
# x02 = np.array([0,np.deg2rad(30),np.deg2rad(90),0,0,0])

x0 = np.zeros(6)
x0[2] = 1

numerical_constants_arm2 = np.array([0.05,  # j0_length [m]
				 0.01,  # j0_com_length [m]
				 4.20,  # j0_mass [kg] 
				 0.001,  # NOT USED j0_inertia [kg*m^2]
				 0.164,  # j1_length [m]
				 0.08,  # j1_com_length [m]
				 1.81,  # j1_mass [kg]
				 0.001,  # NOT USED j1_inertia [kg*m^2]
				 0.158,  # j2_com_length [m]
				 2.259,  # j2_mass [kg]
				 0.001,  # NOT USED j2_inertia [kg*m^2]
				 9.81, # acceleration due to gravity [m/s^2]
				 1, # static friction coeffs
				 1,
				 1,  
				 0.75, #kinetic friction coeffs
				 1,
				 1.25,
				 0.0125, #viscous damping coeffs
				 0.0125,
				 0.0125] 
				) 
#generate solution vectors
if presolved is False:
	runLen = 150
	y1 = np.zeros([runLen,3])
	y2 = np.zeros([runLen,3])

	#go step by step (waiting for input forces)
	if inputForces:
		print("doing this one step at a time")
		for t in range(runLen):
			#arm 1 is "actual" system, arm 2 is where we guess params
			nextStates1 = sp.predict(x0 = x01, dt = dt, numPts = 2)[1]
			nextStates2 = sp.predict(x0 = x02, dt = dt, numPts = 2 , numerical_constants = numerical_constants_arm2)[1]
			x01 = nextStates1
			x02 = nextStates2
			y1[t] = nextStates1[:3]
			y2[t] = nextStates2[:3]
			# print(nextStates[:3])
			print("step ", t, " of ", runLen)

	#generate entire vector at once (only calls odeint once)
	else:
		print("generating entire soln vector")
		sp.x0 = x0
		print("x01 = ", x0)
		print("sp.x0 = ", sp.x0)
		# y1 = sp1.predict(x0 = x01, dt = dt, numPts = runLen)
		y1 = sp.predict()
		print("y1 done")
		print(y1)
		# sp.x0 = x0
		sp.numerical_constants = numerical_constants_arm2
		# y2 = sp2.predict(x0 = x02, dt = dt, numPts = runLen, numerical_constants = numerical_constants_arm2)
		y2 = sp.predict()
		print("y2 done")
	np.save("path_y1", y1)
	np.save("path_y2", y2)

#path already generated
else:
	y1 = np.load("path_y1.npy")
	y2 = np.load("path_y2.npy")
	runLen = len(y2)

while True:
	for i in range(runLen):

		#ARM 1
		# get elbow position
		xElb1 =  0.5 * np.sin(y1[i,0])*np.sin(y1[i,1])
		zElb1 =  0.5 * np.cos((y1[i,1])) 
		yElb1 =   0.5 * np.cos(y1[i,0])*np.sin(y1[i,1])
		# link2RotEff = link1Rot + link2Rot
		xHan1 = xElb1 +  0.5 * np.sin(y1[i,0])*np.sin((y1[i,1] + y1[i,2]))
		zHan1 = zElb1 +  0.5 * np.cos(((y1[i,1] + y1[i,2]))) 
		yHan1 = yElb1 +  0.5 * np.cos(y1[i,0])*np.sin((y1[i,1] + y1[i,2]))

		#ARM 2
		# get elbow position
		xElb2 =  0.5 * np.sin(y2[i,0])*np.sin(y2[i,1])
		zElb2 =  0.5 * np.cos((y2[i,1])) 
		yElb2 =   0.5 * np.cos(y2[i,0])*np.sin(y2[i,1])
		# link2RotEff = link1Rot + link2Rot
		xHan2 = xElb2 +  0.5 * np.sin(y2[i,0])*np.sin((y2[i,1] + y2[i,2]))
		zHan2 = zElb2 +  0.5 * np.cos(((y2[i,1] + y2[i,2]))) 
		yHan2 = yElb2 +  0.5 * np.cos(y2[i,0])*np.sin((y2[i,1] + y2[i,2]))


		try:
			elbow1.remove()
			hand1.remove()
			armline1.remove()
			elbow2.remove()
			hand2.remove()
			armline2.remove()
		except:
			pass
		elbow1, = plt.plot([xElb1-0.5],[yElb1],[zElb1],'bo', markersize = 8)
		armline1, = plt.plot([-0.5,xElb1-0.5,xHan1-0.5],[0,yElb1,yHan1],[0,zElb1,zHan1], 'b-', lw = 6)
		hand1, = plt.plot([xHan1-0.5],[yHan1],[zHan1],'bo', markersize = 8)

		elbow2, = plt.plot([xElb2+ 0.5],[yElb2],[zElb2],'go', markersize = 8)
		armline2, = plt.plot([0.5,xElb2+0.5,xHan2+0.5],[0,yElb2,yHan2],[0,zElb2,zHan2], 'g-', lw = 6)
		hand2, = plt.plot([xHan2+0.5],[yHan2],[zHan2],'go', markersize = 8)


		plt.draw()
		plt.pause(dt*2)
		# plt.cla()

plt.pause(5)