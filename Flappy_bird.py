import gym
import flappy_bird_gym
from stable_baselines3 import DQN
import time
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

#定义环境、DQN agent
env = gym.make('ALE/Breakout-v5')
#
# #如果需要并行环境
#env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
model =DQN('MlpPolicy', env, verbose=1)
# model = DQN.load("DQN_FlappyBird", env=env)
#agent的学习
model.learn(total_timesteps=int(100000))
#agent内参数的保存，在当前目录下多了一个dqn_cartpole.zip文件
model.save("DQN_Breakout-v4")
del model

# model = DQN.load("DQN_FlappyBird", env=env)
# #
# obs = env.reset()
# total_reward = 0
# count = 0
# while True:
#     # Next action:
#     # (feed the observation to your agent here)
#     action, _states = model.predict(obs, deterministic=True)
#
#     # Processing:
#     obs, reward, done, info = env.step(action)
#     total_reward+=reward
#     # Rendering the game:
#     # (remove this two lines during training)
#     # env.render()
#     # time.sleep(1 / 30)  # FPS
#
#     # Checking if the player is still alive
#     if done:
#         break
#     count+=1
# print(total_reward)
# env.close()