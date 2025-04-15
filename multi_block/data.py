import random
import numpy as np
import torch
import torch.nn.functional as F

def generate_data(require_info,goods_low, goods_high,aisle_num,cross_num):
    num_points = random.randint(goods_low, goods_high)  # 随机在goods_low, goods_high范围生成订单数
    # print(num_points)
    orders = random.sample(require_info[7][1:].tolist(), num_points)  # 从require_info[7]里面随机抽取相应数量的订单index
    orders_info = require_info[:, orders]  # 把这些点对应的信息取出来

    order_aisles = orders_info[3]  # 取出来所在巷道
    order_index = orders_info[7]  # 取出来他们的index
    sorted_order_index_index = np.argsort(order_index)  # 按require_info[7]升序排列后对应的索引
    sorted_order_aisles = order_aisles[sorted_order_index_index]  # 获取排序后所在巷道
    sorted_order_index = order_index[sorted_order_index_index]  # 将require_info[7]升序排列

    _, group_start = np.unique(sorted_order_aisles, return_index=True)  # 获取每个巷道开始的位置
    group_sizes = np.diff(np.append(group_start, len(sorted_order_aisles)))  # 每个巷道里有几个点

    order_after_delete = np.array([])  # 存最终删减完的数据

    neighbor_nodes = {}
    for i, (start, size) in enumerate(zip(group_start, group_sizes)):  # 开始位置 包含拣货点数量
        if i == len(group_start) - 1:  # 如果是最后一个巷道
            goods_in_aisle = sorted_order_index[group_start[i]:]  # 取出来这个巷道里所有的货物
        else:
            goods_in_aisle = sorted_order_index[group_start[i]:group_start[i + 1]]  # 取出来巷道内的货物
        if size > 3:
            min_index = np.argmin(goods_in_aisle)  # 最上端拣货点
            max_index = np.argmax(goods_in_aisle)  # 最下端拣货点
            diffs = np.abs(np.diff(goods_in_aisle))  # 巷道里不同idx的差值
            max_diff_idx = np.argmax(diffs)  # 最大的差值所在的位置
            max_diff_indices = [max_diff_idx, max_diff_idx + 1]  # 最大差值的两个点在goods_in_aisle里的idx
            result_indices = np.unique([min_index, max_index] + max_diff_indices)  # 存储这四个点在goods_in_aisle里的idx
            order_after_delete = np.append(order_after_delete, goods_in_aisle[result_indices]).astype(int)  # 把删完的点加进去
            if (goods_in_aisle[min_index]!=goods_in_aisle[max_diff_idx]):  # 如果没有重合 互为邻居
                neighbor_nodes[goods_in_aisle[min_index]] = goods_in_aisle[max_diff_idx]
                neighbor_nodes[goods_in_aisle[max_diff_idx]] = goods_in_aisle[min_index]
            if goods_in_aisle[max_index]!=goods_in_aisle[max_diff_idx+1]:  # 如果没有重合 互为邻居
                neighbor_nodes[goods_in_aisle[max_index]] = goods_in_aisle[max_diff_idx+1]
                neighbor_nodes[goods_in_aisle[max_diff_idx+1]] = goods_in_aisle[max_index]
        else:
            order_after_delete = np.append(order_after_delete, goods_in_aisle).astype(int)  # 把删减后的点加入列表中

    order_after_delete = np.insert(order_after_delete,0,0)  # 把起止点加进去
    orders_coords_std = require_info[6][order_after_delete]  # 把这些点对应的坐标取出来
    orders_coords_std = torch.tensor(orders_coords_std.tolist(), device='cuda')  # tensor版坐标
    order_after_delete = torch.tensor(order_after_delete.tolist(), device='cuda')  # tensor版点
    padding_length = 4 * aisle_num * (cross_num-1)+1 - len(orders_coords_std)
    orders_coords_std = F.pad(orders_coords_std, (0, 0, 0, padding_length), "constant", 0)
    order_after_delete = F.pad(order_after_delete, (0, padding_length), "constant", 0)

    return {'order_after_delete': order_after_delete,
            'orders_coords_std': orders_coords_std,
            'neighbor_nodes':neighbor_nodes,
            'orders_idx':orders}
