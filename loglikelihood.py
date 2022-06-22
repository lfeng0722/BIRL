import numpy as np
import math

import torch


def VI_obj(q_value,beta,state_action):
    tool_s = 0
    for i in range(len(state_action)):
        q=q_value(state_action[i][0])[0]
        q_t= beta*q[state_action[i][1]]
        # print('q_t', q_t)
        q_t = torch.exp(q_t)
        q_b= beta * q
        # print('q_b',q_b)
        # for data in q_b:
        #
        #     data = beta*data
        #     # print('data', data)
        #     # print(torch.exp(beta*q_t))
        q_b = torch.exp(q_b)
        tool_f=sum(q_b)

        # print(torch.exp(q_t),tool_f)
        st = q_t/tool_f
        # print(st, torch.exp(beta*q_t),tool_f)
        log_likelihood =torch.log(st)
        # print(log_likelihood)
        # log_likelihood = beta*q_t-tool_f

        tool_s +=log_likelihood
    # print('loss', tool_s)
    return tool_s