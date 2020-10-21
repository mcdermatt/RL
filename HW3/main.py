from ragdoll import ragdoll
import numpy as np


#TODO:
# take input from policy
# run()
# viz on/off
# fast forward(?)
# check for clipping through ground plane

# torques = np.zeros([5,1000])
torques = np.random.randn(5,1000)


body = ragdoll(viz = True, arms = False, torques = torques)
body.run()