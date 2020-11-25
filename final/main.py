from statePredictor import statePredictor
import numpy as np

#main script
#generates policy to determine friction parameters of robot

#TODO optimize dt
dt = 0.1

#init state predictors
#grond truth model
gt = statePredictor()
gt.dt = dt #length of time between initial and final states
gt.numPts = 2 #only care about start and next state, no need for anything else

#estimated friction model
ef = statePredictor()
ef.dt = dt
ef.numPts = 2

trials = 1000
for trial in range(trials):

	#get random initial states
	a = np.random.rand(3)
	gt.x0[:3] = a 
	ef.x0[:3] = a

	#use actor to determine actions based on states

	#plug state-action pair into critic network to get error

	#use actions chosen by actor network into enviornment

	#get error from critic network

	#calculate ground truth solution
	gtStates = gt.predict()[1]

	#get error from enviornment
	err = np.sum((gtStates - efStates)**2)

	#update actor network

	#update critic network


#save values for actor loss, critic loss, rewards after each trial
	#potentially plot these values as simulation runs??

#save policy -> generate lookup table???
			 # -> MAKE FUNCTION FROM POLICY??? -> add to EOM??