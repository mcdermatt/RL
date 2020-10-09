import time
import numpy as np

a = []
start = time.time()

for _ in range(500):

	np.append(a,np.zeros(6),axis = 0)

finish = time.time()

print("time elapsed = %f" %(finish - start))