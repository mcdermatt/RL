import numpy as np
import copy
import random

class OUNoise:
    """Ornstein-Uhlenbeck process."""
    # mu = 0 -> first moment = mean
    # theta = 0.4
    # sigma = 0.2

    def __init__(self, size, seed = 1, mu=0., theta=0.4, sigma=0.02):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        # self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        #not normally distributed random: sus
        # dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state