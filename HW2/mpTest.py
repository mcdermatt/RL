import multiprocessing 
import time

def do_something(seconds):
	print("sleeping for %f seconds..." %seconds)
	time.sleep(seconds)
	print('Done Sleeping')



if __name__ == "__main__":

	start = time.perf_counter()

	processes = []

	for _ in range(10):
		p = multiprocessing.Process(target=do_something, args = [1.5])
		p.start()
		processes.append(p)

	for process in processes:
		process.join()


	finish = time.perf_counter()

	print("finished in %f seconds" %(finish - start))