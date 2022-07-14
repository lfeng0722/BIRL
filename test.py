import gym


env = gym.make('Breakout-ram-v4')

# stacked gray-scale image
env.reset() # (4, 84, 84)
