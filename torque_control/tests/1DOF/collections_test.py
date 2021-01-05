import collections
import time

#collections are more efficient than numpy arrays for appending and stacking/ popping
#	LESS EFFICIENT AT VECTORIZED OPERATIONS

#describes how experiences will be named
experience = collections.namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

#single trial with state = 1, action = 2, etc.
trial = experience(1,2,3,4,5)

a = collections.deque(maxlen = 100000)
a.append(trial)

start = time.time()

for i in range(100000):
	trial = experience(i,i,i,i,i)
	a.append(trial)

fin = time.time()

print("took ", fin-start, "  seconds")
# print(a)

#get index of experiences (tuples) with a given state of 95 (arbitrarily chosen)
index = [i.state for i in a].index(95)
print(index)
print(a[index])