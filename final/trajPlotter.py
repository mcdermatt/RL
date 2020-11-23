import matplotlib.pyplot as plt
import numpy as np 
import matplotlib
import matplotlib.patches as mpatches

class trajPlotter():
	"""takes in two arm paths and generates graph"""

	filename1 = "C:/Users/Matt/comp138/final/path_y1.npy"
	path1 = np.load(filename1)
	filename2 = "C:/Users/Matt/comp138/final/path_y2.npy"
	path2 = np.load(filename2)

	def __init__(self, path1 = path1, path2 = path2):
		font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 14}
		matplotlib.rc('font', **font)

		t = np.arange(len(path1))

		plt.xlabel('time')
		plt.ylabel('theta')

		fig = plt.figure()
		# ax1 = plt.axes(frameon=False)
		ax1 = fig.add_subplot(frameon = False)
		ax1.set_ylim([-2*np.pi,2*np.pi])
		ax1.axes.get_xaxis().set_visible(False)
		ax1.set_ylabel("theta")
		fig.patch.set_facecolor((.25,.25,.25,0.0))


		j0_patch = mpatches.Patch(color='red', label='Joint 0')
		j1_patch = mpatches.Patch(color='green', label='Joint 1')
		j2_patch = mpatches.Patch(color='blue', label='Joint 2')

		plt.legend(handles=[j0_patch, j1_patch, j2_patch])

		plt.plot(t,path1[:,0], 'r--')
		plt.plot(t,path1[:,1], 'g--')
		plt.plot(t,path1[:,2], 'b--')
		plt.plot(t,path2[:,0], 'r-')
		plt.plot(t,path2[:,1], 'g-')
		plt.plot(t,path2[:,2], 'b-')

		plt.savefig("C:/Users/Matt/comp138/final/pathFig.png", transparent = True)
		# plt.show()