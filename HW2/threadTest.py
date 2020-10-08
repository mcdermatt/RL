import threading
import time

#no threading
def test1():
	print("sleeping for 1 second...")
	time.sleep(1)
	print('Done Sleeping')



if __name__ == "__main__":

	start = time.perf_counter()
	test1()
	finish = time.perf_counter()

	print("finished in %f seconds" %(finish - start))