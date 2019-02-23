import numpy as np
from matplotlib import pyplot

episode = range(500)
reward_list = np.load('reward.npy')
pyplot.plot(episode, reward_list)
pyplot.show()
