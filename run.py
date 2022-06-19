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
import time
from multiprocessing.dummy import Pool as ThreadPool
import torch.utils.data as Data

class svpg_reinforce(object):
    def __init__(self, envs, gamma, alpha, sa_pair, sq_pair_q, s_dim, a_dim, learning_rate, iter, episode, render, temperature, max_episode_length=300):
        self.envs = envs
        self.alpha = alpha
        self.iter = iter
        self.sa_pair = sa_pair
        self.sq_pair_q = sq_pair_q
        self.num_agent = len(self.envs)
        self.observation_dim = s_dim
        self.action_dim = a_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.episode = episode
        self.render = render
        self.temperature = temperature
        self.eps = np.finfo(np.float32).eps.item()
        self.policies = [model(self.observation_dim ,64,3) for _ in range(self.num_agent)]
        self.Q_network = model(self.observation_dim ,64,3)
        self.optimizers = [torch.optim.Adam(self.policies[i].parameters(), lr=self.learning_rate) for i in range(self.num_agent)]
        self.optimizers_Q = torch.optim.Adam(self.Q_network.parameters(), lr=self.learning_rate)
        # self.schedule = torch.optim.lr_scheduler.StepLR(self.optimizers_Q, step_size=3, gamma=0.5)
        self.weight_reward = None
        self.max_episode_length = max_episode_length


    def train(self, const):
        policy_grads = []
        parameters = []
        # print(type(const[1]))
        for i in range(self.num_agent):
            agent_policy_grad = []
            # time_start = time.time()
            # a = [aa.detach() for aa in const[i]]
            for a in const[i]:
                agent_policy_grad.append(-2 * self.policies[i](self.sq_pair_q[0][0]) *a)
            # def process(item):
            #     # print('正在并行for循环')
            #     const = self.policies[i](item[0]).detach()[0][item[1]] - self.Q_network(item[0]).detach()[0][item[1]]
            #     agent_policy_grad.append(-2*self.policies[i](item[0])[0][item[1]]*const)
            #     # time.sleep(5)8
            #
            # items = self.sq_pair_q
            #
            # pool = ThreadPool()
            # pool.map(process, items)
            # for sa in self.sq_pair_q:
            #     const =  self.policies[i](sa[0]).detach()[0][sa[1]] -self.Q_network(sa[0]).detach()[0][sa[1]]
            #     agent_policy_grad.append(-2*self.policies[i](sa[0])[0][sa[1]]*const)
            # time_end = time.time()
            # print('花费时间', time_end - time_start)  # 此处单位为秒
            self.optimizers[i].zero_grad()

            policy_grad = torch.cat(agent_policy_grad).sum()

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


    def run(self):

        for i_iter in range(self.iter):




            batch_split = func(self.sq_pair_q, 64)
            total_loss = 0
            # for _ in tqdm(range(10)):
            for batch_sample in tqdm(batch_split):
                number_batch = 0
                constrain_loss_2 = 0
                const_loss = []
                batch_loss=0
                for i in range(self.num_agent):
                    const_loss_1 = []
                    for j in range(len(batch_sample)-1):
                        constrain_loss = self.policies[i](batch_sample[j][0]).detach()[0][batch_sample[j][1]]-(self.Q_network(batch_sample[j][0])[0][batch_sample[j][1]]-self.gamma * self.Q_network(batch_sample[j+1][0])[0][batch_sample[j+1][1]])
                        const_loss_1.append(constrain_loss.item())
                        # print(self.policies[i](batch_sample[j][0]).detach()[0][batch_sample[j][1]])
                        constrain_loss_2+= abs(constrain_loss)
                    const_loss.append(const_loss_1)
                constrain_loss_1= constrain_loss_2 /self.num_agent

                for _ in range(self.episode):
                    self.train(const_loss)
                # print(self.policies[i](batch_sample[j][0]).detach()[0][batch_sample[j][1]],self.Q_network(batch_sample[j][0])[0][batch_sample[j][1]]-self.gamma*self.Q_network(batch_sample[j+1][0])[0][batch_sample[j+1][1]])
                loss = -VI_obj(self.Q_network,30,batch_sample) +self.alpha * constrain_loss_1
                self.optimizers_Q.zero_grad()
                loss.backward()
                self.optimizers_Q.step()
                # self.schedule.step()
                batch_loss+=loss.item()
                total_loss+=loss.item()
                number_batch+=1
                print(batch_loss)
            total_loss = total_loss/number_batch

            total_reward = 0
            for i, env in enumerate(self.envs):
                obs = env.reset()
                game_reward=0
                count = 0
                while count < self.max_episode_length:
                    if self.render:
                        env.render()
                    # action = np.argmax(self.policies[i](obs).detach().numpy())
                    count += 1
                    action_value = self.policies[i](obs)
                    action = torch.max(action_value,0)[1].data.numpy()
                    # print(action)
                    next_obs, reward, done, info = env.step(action)
                    obs = next_obs
                    game_reward+= reward
                    if done:
                        break

                total_reward+=game_reward
            total_reward=total_reward/(self.num_agent)
            print('iteration %d, loss: %f, reward: %f' % (i_iter, total_loss, total_reward))




if __name__ == '__main__':
    num_agent = 10
    alpha = 10
    sa_pair, sq_pair_q, a_dim, s_dim = load_SVGD_data(num_trajs=6)

    envs = [gym.make('Acrobot-v1') for _ in range(num_agent)]
    envs = [env.unwrapped for env in envs]
    test = svpg_reinforce(envs, gamma=0.99, alpha= alpha, sa_pair=sa_pair, sq_pair_q= sq_pair_q, s_dim=s_dim, a_dim=a_dim, learning_rate=1e-3, iter = 30, episode=10, render=True, temperature=10.0)
    test.run()