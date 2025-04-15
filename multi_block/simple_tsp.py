import numpy as np
import torch

def am_tsp(model, info, file, distance_matrix=None):

    model.eval()

    def make_oracle(model, info, temperature=1.0):
        xy = info['orders_coords_std']
        num_nodes = len(xy)
        info['orders_coords_std']=info['orders_coords_std'].unsqueeze(dim=0)

        with torch.no_grad():  # Inference only
            embeddings, _ = model.embedder(model._init_embed(info))
            fixed = model._precompute(embeddings)

        def oracle(tour):
            with torch.no_grad():
                tour = torch.tensor(tour).long()
                if len(tour) == 0:
                    step_context = model.W_placeholder
                else:
                    step_context = torch.cat((embeddings[0, tour[0]], embeddings[0, tour[-1]]), -1)

                # Compute query = context node embedding, add batch and step dimensions (both 1)
                query = fixed.context_node_projected + model.project_step_context(step_context[None, None, :])

                # Create the mask and convert to bool depending on PyTorch version
                mask = torch.zeros(num_nodes, dtype=torch.uint8) > 0
                mask[tour] = 1
                mask = mask[None, None, :]  # Add batch and step dimension

                log_p, _ = model._one_to_many_logits(query, fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key, mask)
                p = torch.softmax(log_p / temperature, -1)[0, 0]
                assert (p[tour] == 0).all()
                assert (p.sum() - 1).abs() < 1e-5
            return p.tolist()

        return oracle

    oracle = make_oracle(model, info)

    xy = info['order_after_delete']
    tour = []

    while (len(tour) < len(xy)):
        p = oracle(tour)
        p = np.array(p)
        i = np.argmax(p)
        tour.append(i)
        neighbor_nodes = info['neighbor_nodes']
        if info['order_after_delete'][i].item() in neighbor_nodes:
            if info['order_after_delete'].tolist().index(neighbor_nodes[info['order_after_delete'][i].item()]) not in tour:
                # 这里是取到邻接节点重新编号后的idx
                tour.append(info['order_after_delete'].tolist().index(neighbor_nodes[info['order_after_delete'][i].item()]))
    tour.append(tour[0])
    tensor_tour = torch.tensor(tour, device='cuda')
    d = info['order_after_delete'].unsqueeze(0).gather(1, tensor_tour.unsqueeze(0))
    required_info = np.load(f'warehouse_data_4_5/required_info.npy', allow_pickle=True)
    coords_before_delete_x = required_info[1]
    coords_before_delete_y = required_info[2]
    d_np = d.cpu().numpy().flatten()
    coords_before_delete_of_d_x = coords_before_delete_x[d_np]
    coords_before_delete_of_d_y = coords_before_delete_y[d_np]

    next_d = d[:, 1:]
    prev_d = d[:, :-1]
    if distance_matrix is None:
        distance_matrix = np.load('warehouse_data_4_5/distance_matrix.npy', allow_pickle=True)
    l = 0
    prev_d_item = prev_d[0].tolist()
    next_d_item = next_d[0].tolist()
    for i in range(len(prev_d_item)):
        distance = distance_matrix[prev_d_item[i]][next_d_item[i]]
        l += distance
    return l, coords_before_delete_of_d_x, coords_before_delete_of_d_y

