import random
import argparse
from typing import Union, Optional
import multiprocessing as mp

import networkx as nx
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

import torch
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_numpy_cpu(*x: torch.Tensor) -> Union[list, torch.Tensor]:
    outputs = []
    for x_ in x:
        if isinstance(x_, torch.Tensor):
            x_ = x_.detach().cpu().numpy()
        outputs.append(x_)

    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
    return dists_dict


def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


def all_pairs_shortest_path_length_parallel(graph, cutoff=None):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    num_workers = int(mp.cpu_count() * 0.8)
    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(single_source_shortest_path_length_range,
            args=(graph, nodes[int(len(nodes)/num_workers*i):int(len(nodes)/num_workers*(i+1))], cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict


def precompute_dist_data(edge_index, num_nodes, approximate=0) -> np.ndarray:
        '''
        Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
        :return:
        '''
        graph = nx.Graph()
        edge_list = edge_index.transpose(1,0).tolist()
        graph.add_edges_from(edge_list)

        n = num_nodes
        dists_array = np.zeros((n, n))
        np.fill_diagonal(dists_array, 1)
        # dists_dict = nx.all_pairs_shortest_path_length(graph,cutoff=approximate if approximate>0 else None)
        # dists_dict = {c[0]: c[1] for c in dists_dict}
        dists_dict = all_pairs_shortest_path_length_parallel(graph,cutoff=approximate if approximate>0 else None)
        for i, node_i in enumerate(graph.nodes()):
            shortest_dist = dists_dict[node_i]
            for j, node_j in enumerate(graph.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist!=-1:
                    # dists_array[i, j] = 1 / (dist + 1)
                    dists_array[node_i, node_j] = 1 / (dist + 1)

        return dists_array


def score_link_prediction(labels, scores):
    labels, scores = to_numpy_cpu(labels, scores)
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def inductive_eval(cmodel, nodes, gt_labels, X, lambdas = (0, 1, 1)):
    # anode_emb = torch.sparse.mm(data.x, cmodel.attr_emb(torch.arange(data.x.shape[1]).to(cmodel.device)))
    test_data = Data(X, None)
    anode_emb = cmodel.attr_emb(test_data)

    first_embs = anode_emb[nodes[:, 0]]

    sec_embs = anode_emb[nodes[:, 1]]
    res = cmodel.attr_layer(first_embs, sec_embs) * lambdas[1]

    node_emb = anode_emb.clone()

    res = res + cmodel.inter_layer(first_embs, node_emb[nodes[:, 1]]) * lambdas[2]
    
    if len(res.shape) > 1:
        res = res.softmax(dim=1)[:, 1]

    res = res.detach().cpu().numpy()
    return score_link_prediction(gt_labels, res)


def transductive_eval(cmodel, edge_index, gt_labels, data, lambdas=(1, 1, 1)):
    res = cmodel.evaluate(edge_index, data, lambdas)
    if len(res.shape) > 1:
        res = res.softmax(dim=1)[:, 1]

    return score_link_prediction(gt_labels, res)


def detailed_eval(model,test_data,gt_labels,sp_M, evaluate,nodes_keep=None, verbose=False, lambdas=(1,1,1)):
    setting = {}

    setting['Full '] = lambdas
    setting['Inter'] = (0,0,1)
    if lambdas[1]:
        setting['Attr '] = (0,1,0)
    if lambdas[0]:
        setting['Node '] = (1,0,0)
    
    res = {}
    for s in setting:
        if not nodes_keep is None:
            if s != 'Node ':
                res[s] = evaluate(model, test_data, gt_labels,sp_M,nodes_keep,setting[s])
                if verbose:
                    print(s+' ROC-AUC:%.4f AP:%.4f'%res[s])
        else:            
            res[s] = evaluate(model, test_data, gt_labels,sp_M,setting[s])
            if verbose:
                print(s+' ROC-AUC:%.4f AP:%.4f'%res[s])
    return res


def seed_everything(seed: Optional[int] = None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

