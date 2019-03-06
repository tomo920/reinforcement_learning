import numpy as np
import random

theta = 0.15
mu = 0.0
dt = 0.5
sigma = 0.3

class OUnoise:
    def __init__(self, action_size, action_high, action_low):
        initial_noise = []
        for i in range(action_size):
            if random.randint(0, 1) == 0:
                noise = action_low[i]
            else:
                noise = action_high[i]
            initial_noise.append(noise)
        self.noise = np.array(initial_noise)
        self.action_size = action_size

    def _noise(self):
        dx = theta*(mu-self.noise)*dt+sigma*np.random.normal(0.0, np.sqrt(dt), self.action_size)
        self.noise = self.noise + dx
        return self.noise
