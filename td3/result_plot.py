import numpy as np
from matplotlib import pyplot

episode = range(500)
reward_list = np.load('reward.npy')
print(np.sum(reward_list[200:])/500.0)
pyplot.plot(episode, reward_list)
pyplot.show()
