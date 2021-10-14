import time
import types


from tqdm import tqdm
from utils import *
from model import *

from args import *


import torch
from torch.nn import functional as F
from torch_geometric.datasets import Planetoid, CoraFull
from torch_geometric.data import Data
from torch_geometric.utils.to_dense_adj import to_dense_adj
import torch_geometric.transforms as T


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# http://tkipf.github.io/graph-convolutional-networks/
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    # T.RandomNodeSplit(num_val=0.1, num_test=0.2),
    T.RandomLinkSplit(num_val=0.15, num_test=0.15, is_undirected=True,
                      add_negative_train_samples=False),
    T.RandomNodeSplit(),

])

dataset = CoraFull(root='/tmp/CoraFull', transform=transform)

train_data, val_data, test_data = dataset[0]

# dataset = Planetoid(root='/tmp/Cora', name='cora', transform=transform)

# data = dataset[0].to(device)

from model import DEAL, MLP, Emb


# args
args = types.SimpleNamespace(
    gpu=False,
    inductive=True,
    train_ratio=0.5,
    output_dim=64,
    # dataset='Cora',  # Cora...
    dataset='citeseer',  # Cora...
    layer_num=2,
    lr=1e-2,
    repeat_num=1,
    loss='default',
    epoch_num=5000,
    epoch_log=2,
    task='link',
    train_mode='cos',
    BCE_mode=True,
    gamma=2,
    use_tight_alignment=True,
    attr_model='Emb'
)

deal_model = DEAL(
    emb_dim=64,
    attr_num=train_data.x.shape[1],
    node_num=train_data.x.shape[0],
    device=device,
    args=args,
    attr_emb_model=Emb,
)


optimizer = torch.optim.Adam(deal_model.parameters(), lr=0.01, weight_decay=5e-4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Device: using ", device)

day_str = date.today().strftime("%d_%b")

neg_num = 1

# node / attr /  inter
theta_list = (0.1,0.85,0.05)
lambda_list = (0.1,0.85,0.05)

print(f'theta_list:{theta_list}')

A, X, A_train, X_train, data, train_ones, val_edges, test_edges, folder, val_labels, gt_labels, nodes_keep = load_datafile(args)

# A = to_dense_adj(train_data.x.edge_index)

"""
CiteSeer:

Num Nodes: 3327
Num Edges: 4552
Num Attributes: 3703

A shape: %s (3327, 3327)
X shape: %s (3327, 3703)
A_train shape: %s (2824, 2824)
X_train shape: %s (2824, 3703)
X_train shape: %s (2824, 3703)
"""

print('A shape: %s', A.shape)
print('X shape: %s', X.shape)
print('A_train shape: %s', A_train.shape)
print('X_train shape: %s', X_train.shape)
# print('Labels: %s', train_ones.shape)



"""

three sparse matrices, which are the adjacency matrix (A), feature matrix (X) and the label information (z) 
"""


if args.inductive:
    sp_X = convert_sSp_tSp(X).to(device).to_dense()
    sp_attrM = convert_sSp_tSp(X_train).to(device)
    val_labels = A_train[val_edges[:, 0], val_edges[:, 1]].A1
else:
    val_labels = A[val_edges[:, 0], val_edges[:, 1]].A1


init_delta = get_delta(np.stack(A_train.nonzero()),A_train)


def get_train_inputs(data,test_edges,val_edges,batch_size,neg_sample_num=10,undirected=True,inductive=False):
    test_mask = (1-torch.eye(data.dists.shape[0])).bool()
    if not inductive:
        test_mask[test_edges[:,0],test_edges[:,1]]=0
        test_mask[val_edges[:,0],val_edges[:,1]]=0
        if undirected:
            test_mask[test_edges[:,1],test_edges[:,0]]=0
            test_mask[val_edges[:,1],val_edges[:,0]]=0
    test_mask=test_mask.to(data.dists.device)
    filter_dists = data.dists * test_mask
    pos = (filter_dists == 0.5).nonzero()
    filter_dists[pos[:,0],pos[:,1]]=0
    pos = pos.cpu().tolist()
    pos_dict = {}
    for i,j in pos:
        pos_dict[i] = pos_dict.get(i,[])+[j]
        
    neg_dict = {}
    neg = (filter_dists>0.12).nonzero().cpu().tolist()
    for i,j in neg:
        neg_dict[i] = neg_dict.get(i,[])+[j]
    nodes = list(pos_dict.keys())
    random.shuffle(nodes)
    inputs = []
    while True:
        for node in nodes:
            tmp_imput = [node, pos_dict[node], random.sample(neg,neg_sample_num) if len(neg)>neg_sample_num else neg]
            inputs.append(tmp_imput)
            if len(inputs) >= batch_size:
                yield np.array(inputs)
                del inputs[:]
        random.shuffle(nodes)


# sparse_train_adj = edge_index2sp_A(train_data.edge_index, len(train_data.x))
# sparse_train_x = sp.csr_matrix(train_data.x)
# sparse_labels = sp.csr_matrix(train_data.y)


data_loader = iter(get_train_data(A_train, int(X_train.shape[0] *args.train_ratio), np.vstack((test_edges,val_edges)), args.inductive)) #,neg_num
inputs, labels = next(data_loader)


result_list = []
margin_dict = {}
margin_pairs = {}
best_state_dict = None

print(args)
for repeat in tqdm(range(args.repeat_num)):
    for d in margin_dict:
        margin_dict[d].append([])

    deal = DEAL(args.output_dim, X_train.shape[1], X_train.shape[0], device, args, locals()[args.attr_model])

    optimizer = torch.optim.Adam(deal.parameters(), lr=args.lr) 

    max_val_score = np.zeros(1)
    val_result = np.zeros(2)

    running_loss = 0.0

    time1 = time.time()

    # for epoch in tqdm(range(args.epoch_num)):
    for epoch in range(args.epoch_num):
        # get the inputs; data is a list of [inputs, labels]
        
        inputs, labels = next(data_loader)
        labels = labels.to(device)

        """
        inputs.shape: torch.Size([1412, 2])
        tensor([[1288,  274],
        [1288, 2735],
        [2742,  431],
        ...,
        [1265, 1532],
        [ 929, 2032],
        [ 929, 1371]])

        labels.shape: torch.Size([1412])
        tensor([0, 1, 0,  ..., 1, 0, 1])
        labels.sum(): tensor(554)
        
        """

        # zero the parameter gradients
        optimizer.zero_grad()

        #
        # forward + backward + optimize
        #

        loss = deal.default_loss(inputs, labels, data, thetas=theta_list, train_num=int(X_train.shape[0] *args.train_ratio)*2)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        b_num = 5
        if epoch% b_num == b_num-1:   
            avg_loss = running_loss / b_num
            """
            val_edges (734, 2)
            """

            val_scores = transductive_eval(deal, val_edges, val_labels,data ,lambdas=lambda_list)
            
            running_loss = 0.0
            val_result = np.vstack((val_result,np.array(val_scores)))
            tmp_max = np.maximum(np.mean(val_scores), max_val_score)
            rprint('[%8d]  val %.4f %.4f' % (epoch + 1, *val_scores))
            if tmp_max > max_val_score:
                max_val_score = tmp_max
                if args.inductive:
                    raise RuntimeError(nodes_keep.shape)
                    final_scores = avg_loss, *inductive_eval(deal, test_edges, gt_labels,sp_X,nodes_keep ,lambdas=lambda_list)
                else:
                    final_scores = avg_loss, *transductive_eval(deal, test_edges, gt_labels,data ,lambdas=lambda_list)
            for tmp_d in margin_dict:
                pairs = margin_pairs[tmp_d]
                margin_dict[tmp_d][repeat].append([deal.node_forward(pairs).mean().item(),deal.attr_forward(pairs,data).mean().item()])
    
    time2 = time.time()
    print()
    print('\033[93mTime used: %.2f\033[0m'%(time2-time1))

    print(f'ROC-AUC:{final_scores[1]:.4f} AP:{final_scores[2]:.4f}')

    # if args.inductive:
    #     print()
    #     print('Evaluate Validation Dataset')
    #     detailed_eval(deal, val_edges, val_labels, sp_X,inductive_eval,nodes_keep)
    #     print()
    #     print('Evaluate Test Dataset')
    #     detailed_eval(deal, test_edges,gt_labels,sp_X,inductive_eval,nodes_keep)
    # else: 
    #     print()
    #     print('Evaluate Validation Dataset')
    #     detailed_eval(deal, val_edges, val_labels, data,transductive_eval,verbose=True)
    #     print()
    #     print('Evaluate Test Dataset')
    #     detailed_eval(deal,test_edges,gt_labels,data,transductive_eval,verbose=True)


