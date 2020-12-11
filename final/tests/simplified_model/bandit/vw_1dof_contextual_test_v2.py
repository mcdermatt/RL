from vowpalwabbit import pyvw
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from time import sleep
from statePredictor import statePredictor
from sweep import sweep
import pandas as pd


fidelity = 10 #number of discrete values to test for each parameter
dof = 1 # (static, kinetic, viscous)
numBandits = fidelity**dof
numTrials = 10 #number of trials per epoch
dt = 0.5 #time between start and stop of each trial
scale = 0.125 #only looking at 1 param

#init simulated arms
gt = statePredictor() #ground truth model
gt.numPts = 2
gt.dt = dt
ef = statePredictor() #estimated friction model
ef.numPts = 2
ef.dt = dt

#init model
vw = pyvw.vw("--cb 4", quiet=True)

#init bandits
bandits = sweep(dof, fidelity)*scale
print(bandits)
rew = np.zeros([numBandits,numTrials])
avgRew = np.zeros([numBandits,numTrials])

#generate test data
train_data = []
for trial in range(numTrials):
	start = np.random.randn(2)
	gt.x0 = start
	ef.x0 = start

	for ban in range(numBandits):
		ef.numerical_constants[-1] = bandits[ban][0]#only looking at viscous
		result = ef.predict()[-1]
		truth = gt.predict()[-1]
		error = 1-np.e**(-abs(result[1]-truth[1]))
		rew[ban,trial] = error
		avgRew[ban,trial] = np.mean(rew[ban,:trial+1])

		entry = {"action": ban, "cost": error, 'feature1':start[0]}#, 'feature2': '', 'feature3': ''}
		train_data.append(entry)
		# print(train_examples[-1])

train_df = pd.DataFrame(train_data)
train_df['index'] = range(1, len(train_df) + 1)
train_df = train_df.set_index("index")

for i in train_df.index:
  action = train_df.loc[i, "action"]
  cost = train_df.loc[i, "cost"]
  # probability = train_df.loc[i, "probability"]
  feature1 = train_df.loc[i, "feature1"]
  # feature2 = train_df.loc[i, "feature2"]
  # feature3 = train_df.loc[i, "feature3"]

  # Construct the example in the required vw format.
  learn_example = str(action) + ":" + str(cost) + " | " + str(feature1) #+ " " + str(feature2) + " " + str(feature3)
  print(learn_example)

  # Here we do the actual learning.
  vw.learn(learn_example)

# test_data = [{'feature1': 'b', 'feature2': 'c', 'feature3': ''},
#             {'feature1': 'a', 'feature2': '', 'feature3': 'b'},
#             {'feature1': 'b', 'feature2': 'b', 'feature3': ''},
#             {'feature1': 'a', 'feature2': '', 'feature3': 'b'}]

test_data = [{'feature1': 0.25},
			{'feature1': -0.125},
			{'feature1': 0.6}]

test_df = pd.DataFrame(test_data)

# Add index to data frame
test_df['index'] = range(1, len(test_df) + 1)
test_df = test_df.set_index("index")

for j in test_df.index:
	feature1 = test_df.loc[j, "feature1"]
	# feature2 = test_df.loc[j, "feature2"]
	# feature3 = test_df.loc[j, "feature3"]

	test_example = "| " + str(feature1) #+ " " + str(feature2) + " " + str(feature3)

	choice = vw.predict(test_example)
	print(j, choice)