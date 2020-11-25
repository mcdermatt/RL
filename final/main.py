

#main script
#generates policy to determine friction parameters of robot


trials = 1000
for trial in range(trials):

	#get random initial states
	
	#use actor to determine actions based on states

	#plug state-action pair into critic network to get error

	#use actions chosen by actor network into enviornment

	#get DT error(?)

	#calculate ground truth solution

	#get other error

	#update networks


#save values for actor loss, critic loss, rewards after each trial
	#potentially plot these values as simulation runs??

#save policy 
