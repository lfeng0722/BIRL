import numpy as np
import math

def VI_obj(q_value,beta,state_action):
    tool_s = 0
    for i in range(len(state_action)):

        q_t= q_value(state_action[i][0])[0][state_action[i][1]]


        q_b= q_value(state_action[i][0])[0]
        for data in q_b:
            data = beta*data
        tool_f=sum(q_b)
        # log_likelihood =math.log(math.exp(beta*q_t)/tool_f)
        log_likelihood = beta*q_t-tool_f
        tool_s +=log_likelihood
    return tool_s