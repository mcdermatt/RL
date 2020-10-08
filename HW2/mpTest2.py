import concurrent.futures 
import time

def do_something(seconds):
	print("sleeping for %f seconds..." %seconds)
	time.sleep(seconds)
	return 'Done Sleeping ... %f seconds' %seconds



if __name__ == "__main__":

	start = time.perf_counter()

	with concurrent.futures.ProcessPoolExecutor() as executor:

		#Method 1
		# f1 = executor.submit(do_something, 1)
		# print(f1.result())

		#Method 2 - returns results ASAP and as completed
		# secs = [5, 4, 3, 2, 1]
		# results = [executor.submit(do_something, sec) for sec in secs] #list comprehension
		# for f in concurrent.futures.as_completed(results):
		# 	print(f.result())

		#Method 3 - returns results in order they were started
		secs = [5, 4, 3, 2, 1]
		results = executor.map(do_something, secs)

		for result in results:
			#exceptions are handled here
			print(result)



	finish = time.perf_counter()

	print("finished in %f seconds" %(finish - start))