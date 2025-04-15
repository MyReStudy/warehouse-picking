import time

import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning
from torch.utils.data import Dataset
import torch
import os
import pickle
import random
import torch.nn.functional as F
import warnings
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

class TSP(object):

    NAME = 'tsp'
    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"
        order_after_delete = dataset['order_after_delete']
        d = order_after_delete.gather(1, pi)
        next_d = d[:, 1:]
        prev_d = d[:, :-1]
        distance_matrix = np.load('warehouse_data_4_5/distance_matrix.npy', allow_pickle=True)
        cost = []
        batch_size = pi.size(0)
        for b in range(batch_size):
            cost_tmp = 0
            prev_d_item = prev_d[b].tolist()
            next_d_item = next_d[b].tolist()
            for i in range(len(prev_d_item)):
                distance = distance_matrix[prev_d_item[i]][next_d_item[i]]
                cost_tmp += distance
            cost_tmp += distance_matrix[next_d_item[-1]][prev_d_item[0]]
            cost.append(float(cost_tmp))
        cost = torch.tensor(cost, device='cuda')
        return cost, None

class TSPDataset:
    pass