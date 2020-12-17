agent1,critic1 - cartesian reward func, static goal = [1,0.5,1.2]

actor1,critic2 - joint space reward func, goal func = [1,0.5,1.2] coverged to ~+0.8 reward
		start at zero velocity in quadrant 1
		trained with:
			LR_Actor, LR_Critic = 0.0001 (slow)
			Batch size = 512
			checkpoint taken at 6000 trials (50 steps each)

