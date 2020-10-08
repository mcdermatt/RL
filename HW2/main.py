from skrrt import road
import concurrent.futures
import numpy as numpy
import multiprocessing

def doot():
	mapFile = "track2.png"
	mapSize = 30
	Map = road(mapFile, mapSize)
	numRuns = 5
	run = 0
	vis = False
	eps = 0.05
	while run < numRuns:
		print("Run #",run)
		Map.evaluate(eps = eps, visual = vis) 
		Map.improve()
		# print(Map.onStart)
		run += 1

#check and see if random seed is selected

if __name__ == '__main__':

	with concurrent.futures.ProcessPoolExecutor() as executor:

		f1 = executor.submit(doot)
		f2 = executor.submit(doot)
		
		# print(f1)