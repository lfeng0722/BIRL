import gym
import d4rl_atari

env = gym.make('breakout-mixed-v0',render_mode='human') # -v{0, 1, 2, 3, 4} for datasets with the other random seeds

# interaction with its environment through dopamine-style Atari wrapper
observation = env.reset() # observation.shape == (84, 84)
# observation, reward, terminal, info = env.step(env.action_space.sample())
#
# # dataset will be automatically downloaded into ~/.d4rl/datasets/[GAME]/[INDEX]/[EPOCH]
# dataset = env.get_dataset()
# dataset['observations'] # observation data in (1000000, 1, 84, 84)
# dataset['actions'] # action data in (1000000,)
# dataset['rewards'] # reward data in (1000000,)
# dataset['terminals'] # terminal flags in (1000000,)