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
import torch.utils.data as Data

class svpg_reinforce(object):
    def __init__(self, envs, gamma, alpha, sa_pair, sq_pair_q, s_dim, a_dim, learning_rate, iter, episode, render, temperature, max_episode_length=1000):
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
        self.policies = [model(self.observation_dim ,64,2) for _ in range(self.num_agent)]
        self.Q_network = model(self.observation_dim ,64,2)
        self.optimizers = [torch.optim.Adam(self.policies[i].parameters(), lr=self.learning_rate) for i in range(self.num_agent)]
        self.optimizers_Q = torch.optim.Adam(self.Q_network.parameters(), lr=self.learning_rate)
        self.schedule = torch.optim.lr_scheduler.StepLR(self.optimizers_Q, step_size=3, gamma=0.5)
        self.weight_reward = None
        self.max_episode_length = max_episode_length


    def train(self):
        policy_grads = []
        parameters = []

        for i in range(self.num_agent):
            agent_policy_grad = []

            for j in range(len(self.sa_pair)-1):
                agent_policy_grad.append( self.policies[i](self.sq_pair_q[j][0])*(self.Q_network(self.sq_pair_q[j][0])[0][self.sq_pair_q[j][1]]-self.gamma*self.Q_network(self.sq_pair_q[j+1][0])[0][self.sq_pair_q[j+1][1]]))
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
            for _ in tqdm(range(self.episode)):
                self.train()

            batch_split = func(self.sq_pair_q, 64)
            total_loss = 0
            for batch_sample in batch_split:

                constrain_loss_2 = 0
                for i in range(self.num_agent):
                    for j in range(len(batch_sample)-1):
                        constrain_loss = self.policies[i](batch_sample[j][0]).detach()[0][batch_sample[j][1]]-(self.Q_network(batch_sample[j][0])[0][batch_sample[j][1]]-self.gamma*self.Q_network(batch_sample[j+1][0])[0][batch_sample[j+1][1]])
                        constrain_loss_2+=constrain_loss
                constrain_loss_1= constrain_loss_2 /self.num_agent
                loss = -VI_obj(self.Q_network,10,batch_sample) + self.alpha * constrain_loss_1

                # train_x = torch.FloatTensor(self.sq_pair_q)
                # train_dataset = Data.TensorDataset(train_x , loss)
                # train_loader = Data.DataLoader(
                #     dataset=train_dataset,
                #     batch_size=64,
                #     shuffle=True
                # )
                #
                #
                self.optimizers_Q.zero_grad()
                # for step, (batch_x, batch_y) in enumerate(train_loader):
                #     batch_x, batch_y = Variable(batch_x), Variable(batch_y)
                loss.backward()
                self.optimizers_Q.step()
                self.schedule.step()
                total_loss+=loss.item()
            total_reward = 0
            for i, env in enumerate(self.envs):
                obs = env.reset()

                while True:
                    if self.render:
                        env.render()
                    action = np.argmax(self.policies[i](obs).detach().numpy())
                    print(action)
                    next_obs, reward, done, info = env.step(action)
                    obs = next_obs
                    total_reward += reward
                    if done:
                        break

                total_reward+=total_reward
                total_reward=total_reward/(i+1)
            print('iteration %d, loss: %f, reward: %f' % (i_iter, total_loss, total_reward))




if __name__ == '__main__':
    num_agent = 10
    alpha = 10
    sa_pair, sq_pair_q, a_dim, s_dim = load_SVGD_data(num_trajs=10)

    envs = [gym.make('CartPole-v1') for _ in range(num_agent)]
    envs = [env.unwrapped for env in envs]
    test = svpg_reinforce(envs, gamma=0.99, alpha= alpha, sa_pair=sa_pair, sq_pair_q= sq_pair_q, s_dim=s_dim, a_dim=a_dim, learning_rate=1e-4, iter = 30, episode=10, render=True, temperature=10.0)
    test.run()