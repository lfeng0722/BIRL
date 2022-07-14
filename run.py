import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from net import model
from svpg import _square_dist, _Kxx_dxKxx, calc_returns
from utils import vector_to_parameters, parameters_to_vector
from load_data import load_SVGD_data
from loglikelihood import VI_obj
import gym
from tqdm import tqdm
from split import func
import random
# import d4rl_atari
import flappy_bird_gym
import time
from multiprocessing.dummy import Pool as ThreadPool
import torch.utils.data as Data




class svpg_reinforce(object):
    def __init__(self, evn, envs, beta,gamma, alpha,  sq_pair_q, s_dim, learning_rate_q, learning_rate, iter, episode, render, temperature, action_space , max_episode_length=500):
        self.evn = evn
        self.envs = envs
        self.beta = beta
        self.alpha = alpha
        self.action_s = action_space
        self.iter = iter
        self.learning_rate_q = learning_rate_q
        self.sq_pair_q = sq_pair_q
        self.num_agent = len(self.envs)
        self.observation_dim = s_dim
        # self.state_set = state_set
        # self.action_set = action_set
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.episode = episode
        self.render = render
        self.temperature = temperature
        self.eps = np.finfo(np.float32).eps.item()
        self.policies = [model(self.observation_dim ,64,self.action_s).cuda() for _ in range(self.num_agent)]
        self.Q_network = model(self.observation_dim ,64,self.action_s).cuda()
        self.optimizers = [torch.optim.Adam(self.policies[i].parameters(), lr=self.learning_rate) for i in range(self.num_agent)]
        self.optimizers_Q = torch.optim.Adam(self.Q_network.parameters(), lr=self.learning_rate_q)
        # self.schedule = torch.optim.lr_scheduler.StepLR(self.optimizers_Q, step_size=3, gamma=0.5)
        self.weight_reward = None
        self.max_episode_length = max_episode_length


    def train(self, Q_value,batch,input):
        #SVGD inference part
        policy_grads = []
        parameters = []
        reward_value_1 = 0

        for i in range(self.num_agent):
            agent_policy_grad = []


            reward_value = self.policies[i](input)

            for j in range(len(Q_value)-1):
                rv=reward_value[j][batch[j][1]]
                reward_value_1+=torch.normal(rv.detach()-Q_value[j])
                agent_policy_grad.append(-2 * self.alpha*rv*(rv.detach()-Q_value[j].detach()) )

            self.optimizers[i].zero_grad()

            policy_grad = torch.cat(agent_policy_grad).sum()
            policy_grad = policy_grad.cuda()
            policy_grad.backward()
            # print(self.policies[i].parameters())
            param_vector, grad_vector = parameters_to_vector(self.policies[i].parameters(), both=True)
            policy_grads.append(grad_vector.unsqueeze(0))
            parameters.append(param_vector.unsqueeze(0))

        parameters = torch.cat(parameters)
        Kxx, dxKxx = _Kxx_dxKxx(parameters, self.num_agent)
        policy_grads = 1. / self.temperature * torch.cat(policy_grads)
        grad_logp = torch.mm(Kxx, policy_grads)
        grad_theta = - (grad_logp + dxKxx) / self.num_agent

        for i in range(self.num_agent):
            vector_to_parameters(grad_theta[i], self.policies[i].parameters(), grad=True)
            self.optimizers[i].step()
            # del self.policies[i].rewards[:]
            # del self.policies[i].log_probs[:]
        return reward_value_1 #gai

    def run(self):
        batch_split = func(self.sq_pair_q, 64)
        for i_iter in tqdm(range(self.iter)):
            #batch split
            total_loss = 0

            for batch_sample in batch_split:
                # number_batch = 0
                # Q_value=[]
                batch_loss=0
                Q_diff=[]
                tool_s = 0
                input = torch.cat( [ batch_sample[i][0] for i in range (len(batch_sample)) ], 0 )
                q_value = self.Q_network(input)
                # start = time.time()
                for j in range(len(batch_sample)):

                    q = q_value[j]

                    q_t = self.beta * q[batch_sample[j][1]]
                    q_t = torch.exp(q_t)
                    q_b = self.beta * q
                    q_b = torch.exp(q_b)
                    tool_f = sum(q_b)
                    st = q_t / tool_f
                    log_likelihood = torch.log(st)
                    tool_s += log_likelihood

                    # Q_value.append(q_value)
                    if j<len(batch_sample)-1:
                        q_diff = q[batch_sample[j][1]] - self.gamma * q[batch_sample[j + 1][1]]
                        Q_diff.append(q_diff)

                # end = time.time()
                # print(end - start)
                for _ in range(self.episode):#run SVGD inference
                    rwd_f = self.train(Q_diff,batch_sample,input)

                constrain_loss_1 = rwd_f/self.num_agent #gai


                loss = -tool_s +self.alpha * constrain_loss_1
                loss = loss.cuda()
                self.optimizers_Q.zero_grad()
                loss.backward()
                self.optimizers_Q.step()

                # batch_loss+=loss.item()
                total_loss+=loss.item()
                # number_batch+=1
            print('loss',total_loss)
        max_reward = []
        for i, env in enumerate(self.envs):
            env.seed(520)
            obs = env.reset()
            game_reward=0
            count = 0
            while count < self.max_episode_length:
                # if env =='breakout-expert-v0' or env == 'pong-expert-v0':
                # obs = obs.flatten()

                count += 1
                action_value = self.policies[i](torch.FloatTensor(obs).cuda())
                action = torch.max(action_value,0)[1].cpu().data.numpy()
                # print(action)
                next_obs, reward, done, info = env.step(action)
                obs = next_obs
                game_reward+= reward
                if done:
                    break
            max_reward.append(game_reward)
        best_policy = np.argmax(max_reward)

        env = gym.make(self.evn)
        # env.seed(520)
        reward_show = []
        for j in range(10):
            obs = env.reset()
            count = 0
            game_reward = 0

            while count < self.max_episode_length:
                # if evn =='breakout-expert-v0' or env == 'pong-expert-v0':
                # obs = obs.flatten()
                if self.render:
                    env.render()

                count += 1
                action_value = self.policies[best_policy](torch.FloatTensor(obs).cuda())
                action = torch.max(action_value,0)[1].cpu().data.numpy()
                # print(action)
                next_obs, reward, done, info = env.step(action)
                obs = next_obs
                game_reward+= reward
                if done:
                    break
            reward_show.append(game_reward)
        env.close()

        print('Mean reward: %f , max reward: %f, min reward: %f, std: %f'% (np.mean(reward_show), np.max(reward_show), np.min(reward_show), np.std(reward_show)))



if __name__ == '__main__':
    num_agent = 1000
    lambda_c = 10
    beta = 50
    action_space = 4 #change action space here for each environment
    evn = "LunarLander-v2"
    sq_pair_q, s_dim = load_SVGD_data(evn ,num_trajs=3)

    # torch.manual_seed(20)
    # np.random.seed(20)
    # random.seed(20)
    envs = [gym.make(evn) for _ in range(num_agent)] #change environment here, and also change the environment in load_data.py
    envs = [env.unwrapped for env in envs]
    test = svpg_reinforce(evn, envs, beta, gamma=0.99, alpha= lambda_c, sq_pair_q= sq_pair_q, s_dim=s_dim, learning_rate_q=1e-3, learning_rate=1e-3, iter = 15, episode=10, render=False, temperature=10, action_space = action_space)
    test.run()