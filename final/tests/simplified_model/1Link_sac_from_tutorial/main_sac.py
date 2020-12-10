import pybullet_envs
import gym
import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve
from gym import wrappers
from statePredictor import statePredictor
import torch

if __name__ == '__main__':

    agent = Agent(n_actions=3)
    numTrials = 5000
    dt = 0.01 # was 0.5
    numSteps = 1 # was 10
    scale = np.array([0.5,0.5,0.25])

    #ground truth friction model
    gt = statePredictor()
    gt.dt = dt/numSteps #length of time between initial and final states
    gt.numPts = 2

    #estimated friction model
    ef = statePredictor()
    ef.dt = dt/numSteps #lol
    ef.numPts = 2

    filename = '1DOF_state_dependant_friction.png'
    figure_file = 'plots/' + filename

    best_score = -100
    score_history = []

    for i in range(numTrials):
        
        # states = torch.randn(2)
        states = np.random.randn(2)
        states[1] = states[1] * 10 #multiplying initial velocity by 10 to make effects of damping and kinetic friction more apparent
        r1 = np.random.rand()
        if r1 > 0.9:
            states[1] = 0 #set starting velocity to zero since random variable never will
        gt.x0 = states 
        ef.x0 = states
        efStates = states #temp
        gtStates = states

        score = 0
        for step in range(numSteps):
            action = agent.choose_action(efStates) #base action on current states of estimation agent
            action = action*scale

            ef.numerical_constants[5:] = action
            ef.x0 = efStates
            efStates_next = ef.predict()[1]
            
            gt.x0 = gtStates
            #make friction parameters function of joint position
            gt.numerical_constants[5] = 0.1 + 0.1*np.sin(gtStates[0]) 
            gt.numerical_constants[6] = 0.125 + 0.125*np.sin(gtStates[0]) 
            gt.numerical_constants[7] = 0.05 + 0.05*np.sin(gtStates[0]) 
            gtStates_next = gt.predict()[1]

            reward = np.e**(-abs(gtStates_next[1]-efStates_next[1])) #vel
            if reward < -100:
                reward = -100

            if step == (numSteps-1):
                done = 1 
            else:
                done = 0
            
            score += reward
            agent.remember(efStates, action, reward, efStates_next, done)
            agent.learn()
            efStates = efStates_next
            gtStates = gtStates_next
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
        agent.save_models()

        if i % 50 == 0:
            print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
            print("action = ", action)


    x = [i+1 for i in range(numTrials)]
    plot_learning_curve(x, score_history, figure_file)