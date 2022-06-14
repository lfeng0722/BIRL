import numpy as np
import torch

#load exper trajectory
def load_SVGD_data(num_trajs=None):

    # path_head = f"volume/{env}/expert_trajs.npy"
    # path = resource_filename("sbirl", path_head)
    data = np.load('CartPole-v1/expert_trajs.npy', allow_pickle=True)



    data_trajs = data.reshape(1)[0]["trajs"]

    if num_trajs is not None:
        data_trajs = data_trajs[:num_trajs]

    state_next_state = []
    state_next_state_q = []
    # action_next_action = []

    s_dim = data_trajs[0][0][0].shape[1]
    a_dim = data_trajs[0][0][0].shape[0]

    for traj in data_trajs:
        for t in range(len(traj) - 1):
            state = traj[t][0]
            action = traj[t][1]
            s_n_s = np.append(state,action)
            # s_n_s = torch.tensor(s_n_s,dtype=torch.float)
            #
            # state = torch.tensor(traj[t][0],dtype=torch.float)
            # action = torch.tensor(traj[t][1],dtype=torch.float)
            s_n_s_q = (state,action)
            state_next_state.append(s_n_s)
            state_next_state_q.append(s_n_s_q)


    return state_next_state, state_next_state_q,a_dim, s_dim