from skrrt import road
import concurrent.futures
import numpy as numpy
import multiprocessing
import time

#TODO
#   Use seed as input for executors?
#	share variable space for different processes

def doot(Map):
	
	numRuns = 5
	run = 0
	vis = False
	eps = 0.05
	while run < numRuns:
		print("Run #",run)
		Map.evaluate(eps = eps, visual = vis) 
		# print(Map.onStart)
		run += 1

if __name__ == '__main__':

	start = time.perf_counter()

	mapFile = "track2.png"
	mapSize = 15
	Map = road(mapFile, mapSize)

	with concurrent.futures.ProcessPoolExecutor() as executor:

		f1 = executor.submit(doot,Map)
		f2 = executor.submit(doot,Map)
		f3 = executor.submit(doot,Map)
		f4 = executor.submit(doot,Map)

		print(Map.test)

	finish = time.perf_counter()
	print("finished in %f seconds" %(finish - start))