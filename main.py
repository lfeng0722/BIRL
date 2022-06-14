import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import math
import torch
import torch.optim as optim
from torch.autograd import Variable
from load_data import load_SVGD_data
from particles import phi


sa_pair,v,a_dim,s_dim = load_SVGD_data(num_trajs=10)



print(sa_pair[1])