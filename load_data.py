import numpy as np
import random
import torch
from split import *
#load exper trajectory
def load_SVGD_data(env,num_trajs):

    # path_head = f"volume/{env}/expert_trajs.npy"
    # path = resource_filename("sbirl", path_head)
    path_head = f"{env}/expert_trajs.npy"
    data = np.load(path_head, allow_pickle=True)

    if env == 'LunarLander-v2' or env=='CartPole-v1' or env =='Acrobot-v1':
        data_trajs = data.reshape(1)[0]["trajs"]

        if num_trajs is not None:
            data_trajs = data_trajs[:num_trajs]

        state_next_state = []
        state_next_state_q = []
        # action_next_action = []
        state_tuple = []
        action_tuple = []

        s_dim = data_trajs[0][0][0].shape[1]


        for traj in data_trajs:
            for t in range(len(traj) - 1):
                state = torch.tensor(traj[t][0]).cuda()
                action = torch.tensor(traj[t][1]).cuda()

                s_n_s_q = (state,action)
                state_tuple.append(state)
                action_tuple.append(action)

                state_next_state_q.append(s_n_s_q)

        return state_next_state_q,s_dim
    elif env == 'FlappyBird-v0':
        # data = func(data,5)
        random.shuffle(data)
        if num_trajs is not None:
            data_trajs = data[:num_trajs*200]
        data_out=[]
        for step in data_trajs:
            s_n_s_q = ([np.array(step[0])], [step[1]])
            data_out.append(s_n_s_q)

        # print(data_trajs[0][0][1].shape)
        # print(data_out[1][0][0])
        return data_out, len(data_trajs[0][0])
    else:
        # random.shuffle(data)
        num_trajs =num_trajs*2000
        data_trajs = data[:num_trajs ]

        data_out = []
        for step in data_trajs:
            step[0] = step[0].flatten()
            s_n_s_q = ([np.array(step[0])], [step[1]])
            print(step[1])
            data_out.append(s_n_s_q)



        return data_out, len(data_trajs[0][0])