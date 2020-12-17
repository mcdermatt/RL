import numpy as np

def FK(theta0, theta1, theta2):

	l1 = 1
	l2 = 1

	d = l2*np.sin(theta1) + l2*np.sin(np.pi - theta1 - theta2)

	x = d*np.cos(theta0)
	y = l1*np.cos(theta1)-l2*np.cos(np.pi-theta1-theta2)
	z = d*np.sin(theta0)

	return np.array([x,y,z])

def IK(x,y,z):
	l1 = 1 # upper arm
	l2 = 1 # lower arm l1+l2=1

	r,phi,theta = cartesian_to_spherical(x,y,z)
	a0,a1,a2 = get_joint_angles(xIn,yIn,zIn)
	xElbow,yElbow,zElbow = get_elbow_pos(xIn,yIn,zIn)
	uwX, uwY, uwZ = get_l3_pos(xIn,yIn,zIn)

	return(a0, a1, a2)


def cartesian_to_spherical(x,y,z):
	r = np.sqrt((x*x)+(y*y)+(z*z))
	phi = np.arctan2((np.sqrt((x  * x) + (y * y))), z )
	theta = np.arctan2(y, x)
	return (r,phi,theta)

def get_joint_angles(x,y,z):
	l1 = 1 # upper arm
	l2 = 1 # lower arm l1+l2=1, easiest if upper and lower arm are same length
	
	(r,phi,theta) = cartesian_to_spherical(x,y,z)
	
	#elbow
	a2 = np.arccos(((l1*l1)+(l2*l2)-(r*r))/(-2*l1*l2))
	#shoulder side to side
	a0 = theta
	#shoulder up down
	a1 = np.pi + phi + np.arccos(((l1*l1)-(l2*l2)+(r*r))/(-2*l1*r))
	
	return(a0,a1,a2)

def get_elbow_pos(x,y,z):

	l1 = 1 # upper arm
	l2 = 1 # lower arm l1+l2=1, easiest if upper and lower arm are same length

	(r, phi, theta) = cartesian_to_spherical(x,y,z)
	(a0,a1,a2) = get_joint_angles(x,y,z)
	
	xElbow = ( l1 * np.cos(a0)*np.sin(a1))
	yElbow = ( l1 * np.sin(a0)*np.sin(a1))
	zElbow = ( l1 * np.cos(a1))
	
	return (xElbow, yElbow, zElbow)

