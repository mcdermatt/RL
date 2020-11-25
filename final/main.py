from statePredictor import statePredictor
import numpy as np

#main script
#generates policy to determine friction parameters of robot


#init CUDA
if torch.cuda.is_available():
	device = torch.device("cuda:0")
	torch.set_default_tensor_type('torch.cuda.FloatTensor') 
	print("Running on GPU")
else:
	device = torch.device("cpu")
	torch.set_default_tensor_type('torch.FloatTensor')
	print("Running on the CPU")


#TODO optimize dt
dt = 0.1

#init state predictors ---------------------
#grond truth model
gt = statePredictor()
gt.dt = dt #length of time between initial and final states
gt.numPts = 2 #only care about start and next state, no need for anything else
#estimated friction model
ef = statePredictor()
ef.dt = dt
ef.numPts = 2

#init NN------------------------------------
agent = Agent(6,9)

trials = 1000
for trial in range(trials):
	print("Trial #: ", trial)

	#get random initial states
	a = np.random.rand(3)
	gt.x0[:3] = a 
	ef.x0[:3] = a

	#use actor to determine actions based on states
	agent.actor.eval()

	with torch.no_grad(): #TODO- verify if I really want this
		action = agent.actor.forward(states.view(-1,6))
	agent.actor.train() #unlocks actor

	#bring back to cpu for running on model - not sure if this is necessary
	act = action.cpu().detach().numpy()

	#plug actions chosen by actor network into enviornment
	ef.numerical_constants[12:] = act #in this case actions are friction parameters
	efStates = ef.predic()[1]

	#plug state-action pair into critic network to get error

	#get output from critic network

	#calculate ground truth solution
	gtStates = gt.predict()[1]

	#get error from enviornment
	err = np.sum((gtStates - efStates)**2)

	#get difference between critic and enviornment

	#updates actor and critic network
	agent.step(state, action, reward, state_next, done)
		#step calls agent.learn()
			#learn() calls critic forward and update
			#TODO- is this sufficient???


#save values for actor loss, critic loss, rewards after each trial
	#potentially plot these values as simulation runs??

#save policy -> generate lookup table???
			 # -> MAKE FUNCTION FROM POLICY??? -> add to EOM??