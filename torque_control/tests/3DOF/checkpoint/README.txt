agent1,critic1 - cartesian reward func, static goal = [1,0.5,1.2]

actor2(100,50),critic2(200,100) - joint space reward func, goal func = [1,0.5,1.2] coverged to ~+0.8 better than it started for each trial (inb4 not a great metric)
		start at zero velocity in quadrant 1
		trained with:
			LR_Actor, LR_Critic = 0.0001 (slow)
			Batch size = 512
			checkpoint taken at 6000 trials (50 steps each)
			fidelity (aka timesteps) = 0.1
		CONVERGES!! - (NOT SUPER WELL THOUGH)
		
actor3(400,200),critic3(400,200)- joint space moving goal
		random starting position and velocity
		LR_Actor, LR_Critic = 0.0001
		batch size = 1024
		(currently training)

