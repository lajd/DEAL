import os
import time
import math
from argparse import Namespace, ArgumentParser
from typing import Tuple, Union

import torch
from torch_geometric.datasets import Planetoid, CoraFull, PPI, Coauthor, Amazon
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch_geometric.utils.negative_sampling import negative_sampling
import torch_geometric.transforms as T

from model import DEAL, get_attr_emb_model_class
from utils import str2bool, inductive_eval, transductive_eval, precompute_dist_data, seed_everything

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


class AugmentedData(Data):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def shuffle(self):
        shuffle_indices = torch.randperm(self.edge_index.shape[1])
        edge_index = self.edge_index[:, shuffle_indices]
        edge_labels = self.edge_labels[shuffle_indices]
        return AugmentedData(
            x=self.x,
            edge_index=edge_index,
            edge_labels=edge_labels,
            dists=self.dists
        )

    @property
    def inputs(self):
        return self.edge_index, self.edge_labels

    def _get_negative_samples(self, shuffle: bool = True, negative_sampling_ratio: float = 1):
        if negative_sampling_ratio > 1:
            num_iterations = math.ceil(negative_sampling_ratio)
        else:
            num_iterations = 1

        neg_edge_indices = []
        for _ in range(num_iterations):
            neg_edge_index_ = negative_sampling(
                edge_index=self.edge_index,
                num_nodes=self.num_nodes,
                num_neg_samples=None,  # Default to 1:1 ratio
                method='sparse'
            )
            neg_edge_indices.append(neg_edge_index_)
        if len(neg_edge_indices) == 1:
            neg_edge_index = neg_edge_indices[0]
        else:
            neg_edge_index = torch.cat(neg_edge_indices, dim=1)
            # Drop duplicate columns
            neg_edge_index = torch.unique(neg_edge_index, dim=1)
            # Sample desired number
            neg_edge_index = neg_edge_index[0: math.ceil(self.num_nodes * negative_sampling_ratio)]

        all_edge_index = torch.cat(
            [self.edge_index, neg_edge_index],
            dim=-1,
        )

        all_edge_label = torch.cat([
            self.edge_labels,
            self.edge_labels.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        if shuffle:
            shuffle_indices = torch.randperm(all_edge_index.shape[1])
            all_edge_index = all_edge_index[:, shuffle_indices]
            all_edge_label = all_edge_label[shuffle_indices]

        return all_edge_index.to(device), all_edge_label.to(device)

    def add_negative_samples(self, shuffle: bool = True, negative_sampling_ratio: float = 1) -> Data:
        all_edge_indices, all_edge_labels = self._get_negative_samples(
            shuffle=shuffle,
            negative_sampling_ratio=negative_sampling_ratio
        )
        data = AugmentedData(
            x=self.x,
            edge_index=all_edge_indices,
            edge_labels=all_edge_labels,
            dists=self.dists if hasattr(self, 'dists') else None
        )
        return data


class InductiveDeal:
    def __init__(self, parsed_arguments: Namespace):
        self.seed = seed_everything(parsed_arguments.seed)
        self.args = parsed_arguments
        self.dataset_name = parsed_arguments.dataset_name.lower()
        self.device = device
        self.train_data, self.val_data, self.test_data, self.data = self.get_data()
        self.model = self._get_model(parsed_arguments, self.train_data).to(self.device)
        # node / attr /  inter
        self.lambda_list = args.thetas  # (0.1, 0.85, 0.05)
        self.lambda_list = args.lambdas  # (0.1, 0.85, 0.05)
        ind_lambdas = list(self.lambda_list)
        ind_lambdas[0] = 0
        self.inductive_lambda_list = ind_lambdas

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=parsed_arguments.lr,
            weight_decay=parsed_arguments.wd
        )
        self.profiler = torch.profiler.profile(record_shapes=True)

    @staticmethod
    def parse_args():
        parser = ArgumentParser()
        parser.add_argument('--seed', dest='seed', default=42, type=int)
        parser.add_argument('--dataset', dest='dataset_name', default='Cora', type=str)
        parser.add_argument('--use_tight_alignment', dest='use_tight_alignment', action='store_true',
                            help='use Strong Alignment', default=False)
        parser.add_argument('--use_loose_alignment', dest='use_tight_alignment', action='store_false',
                            help='use Weak Alignment', default=True)
        parser.add_argument('--transductive_val', dest='transductive_val', action='store_true',
                            help='Perform transductive validation', default=False)
        parser.add_argument('--inductive_val', dest='transductive_val', action='store_false',
                            help='Perform inductive validation', default=True)
        # dataset
        parser.add_argument('--val_ratio', dest='val_ratio', default=0.1, type=float,
                            help='Percentage of data to use for [inductive] validation')
        parser.add_argument('--test_ratio', dest='test_ratio', default=0.1, type=float,
                            help='Percentage of data to use for [inductive] testing')
        parser.add_argument('--dropout', dest='dropout', type=float, default=0.3,
                            help='Dropout probability')
        parser.add_argument('--negative_sampling_ratio_train', dest='negative_sampling_ratio_train', default=1,
                            type=float, help="Negative sampling ratio (as a ratio to the true edges) for training split")
        parser.add_argument('--negative_sampling_ratio_val', dest='negative_sampling_ratio_val', default=1,
                            type=float, help="Negative sampling ratio (as a ratio to the true edges) for val split")
        parser.add_argument('--negative_sampling_ratio_test', dest='negative_sampling_ratio_test', default=1,
                            type=float, help="Negative sampling ratio (as a ratio to the true edges) for test split")
        # Model Hparams
        parser.add_argument('--lambdas', action='store', dest='lambdas',
                            type=str, nargs='*', default=[0.1, 0.85, 0.05],
                            help="Lambda values")
        parser.add_argument('--thetas', action='store', dest='thetas',
                            type=str, nargs='*', default=[0.1, 0.85, 0.05],
                            help="Theta values")
        parser.add_argument('--n_hidden_layers', dest='n_hidden_layers', default=2, type=int)
        parser.add_argument('--feature_dim', dest='feature_dim', default=64, type=int)
        parser.add_argument('--hidden_dim', dest='hidden_dim', default=64, type=int)
        parser.add_argument('--output_dim', dest='output_dim', default=64, type=int)
        parser.add_argument('--lr', dest='lr', default=1e-2, type=float)
        parser.add_argument('--wd', dest='wd', default=0, type=float)
        parser.add_argument('--n_epochs', dest='n_epochs', default=5000, type=int)
        parser.add_argument('--epoch_log', dest='epoch_log', default=0.05, type=Union[int, float])
        parser.add_argument('--gamma', dest='gamma', default=2, type=float)
        parser.add_argument('--approximate', dest='approximate', default=-1, type=int,
                            help='k-hop shortest path distance. -1 means exact shortest path')  # -1, 2
        parser.add_argument('--use_order', dest='use_order', default=False, type=str2bool,
                            help='whether use Order Strategy, default False')
        parser.add_argument('--train_mode', dest='train_mode', default='cos', type=str,
                            help='cos, dot, all, pdist, default cos')
        parser.add_argument('--loss', dest='loss', default='default', type=str,
                            help='loss function options: default, etc.')
        parser.add_argument('--attr_model', dest='attr_model', default='Emb', type=str,
                            help='Attribute embedding model, Emb, SAGE, GAT ... , default Emb')
        parser.add_argument('--bce', dest='BCE_mode', default=True, type=str2bool, help='If use BCE_mode, default True')
        parsed_args = parser.parse_args()
        return parsed_args

    @staticmethod
    def _get_model(parsed_args: Namespace, data: Data):

        emb_model = get_attr_emb_model_class(parsed_args.attr_model.lower())

        deal_model = DEAL(
            emb_dim=parsed_args.output_dim,
            attr_num=data.x.shape[1],
            node_num=data.x.shape[0],
            attr_emb_model=emb_model,
            n_hidden_layers=parsed_args.n_hidden_layers,
            feature_dim=parsed_args.feature_dim,
            hidden_dim=parsed_args.hidden_dim,
            train_mode=parsed_args.train_mode,
            BCE_mode=parsed_args.BCE_mode,
            gamma=parsed_args.gamma,
            use_tight_alignment=parsed_args.use_tight_alignment,
            dropout_p=parsed_args.dropout,
        )
        return deal_model

    @staticmethod
    def show_dataset_info(name: str, data: Data) -> None:
        # Assume single data object
        print(f'---------- {name} Data Information----------')
        print(data)

    def get_data(self) -> Tuple[Data, Data, Data, Data]:
        # Transform, extracting train/val/test split of nodes
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(self.device),
            T.RandomNodeSplit(num_val=self.args.val_ratio, num_test=self.args.test_ratio),
        ])
        if self.dataset_name == 'cora':
            dataset = Planetoid(root='/tmp/Cora', name='cora', transform=transform)
        elif self.dataset_name == 'corafull':
            dataset = CoraFull(root='/tmp/CoraFull', transform=transform)
        elif self.dataset_name == 'citeseer':
            dataset = Planetoid(root='/tmp/CiteSeer', transform=transform, name='CiteSeer')
        elif self.dataset_name == 'pubmed':
            dataset = Planetoid(root='/tmp/PubMed', transform=transform, name='pubmed')
        elif self.dataset_name == 'coauthor-cs':
            dataset = Coauthor(root='/tmp/CoauthorCS', transform=transform, name='CS')
        elif self.dataset_name == 'coauthor-physics':
            dataset = Coauthor(root='/tmp/CoauthorPhysics', transform=transform, name='Phsyics')
        elif self.dataset_name == 'amazon-computers':
            dataset = Amazon(root='/tmp/amazon-computers', transform=transform, name='computers')
        elif self.dataset_name == 'amazon-photos':
            dataset = Amazon(root='/tmp/amazon-photos', transform=transform, name='photo')
        else:
            raise ValueError(f'Unsupported data type `{self.dataset_name}`')

        data = dataset[0]
        InductiveDeal.show_dataset_info("All data", data)
        # Extract train/val/test splits for the inductive link prediction setting
        # Nodes/edges in val/test split are not contained in the training set

        if self.args.transductive_val:
            train_val_mask = data.train_mask | data.val_mask
            # Training data contains both train_mask and val_mask nodes/edges
            train_edge_index, train_edge_attr = subgraph(train_val_mask, edge_index=data.edge_index, relabel_nodes=True)

            # The validation set is a subset of the training set. There are no links to the rest of
            # the training set
            val_edge_index, val_edge_attr = subgraph(data.val_mask, edge_index=data.edge_index, relabel_nodes=True)
            #
            train_x = data.x[train_val_mask]
        else:
            train_edge_index, train_edge_attr = subgraph(data.train_mask, edge_index=data.edge_index, relabel_nodes=True)
            val_edge_index, val_edge_attr = subgraph(data.val_mask, edge_index=data.edge_index, relabel_nodes=True)
            #
            train_x = data.x[data.train_mask]

        val_x = data.x[data.val_mask]
        test_x = data.x[data.test_mask]
        test_edge_index, test_edge_attr = subgraph(data.test_mask, edge_index=data.edge_index, relabel_nodes=True)

        # Create Data objects for train/val/test splits.
        # Edge labels are all ones indicating all provided edges are valid
        train_data = Data(
            x=train_x,
            edge_index=train_edge_index,
            edge_labels=torch.ones(train_edge_index.shape[1]).long()
        )

        val_data = Data(
            x=val_x,
            edge_index=val_edge_index,
            edge_labels=torch.ones(val_edge_index.shape[1]).long()
        )

        test_data = AugmentedData(
            x=test_x,
            edge_index=test_edge_index,
            edge_labels=torch.ones(test_edge_index.shape[1]).long()
        )

        InductiveDeal.show_dataset_info("Train data", train_data)
        InductiveDeal.show_dataset_info("Val data", val_data)
        InductiveDeal.show_dataset_info("Test data", test_data)

        return train_data, val_data, test_data, data

    def prepare_data(self, data_split_name: str, data: Data) -> Data:

        save_dir = os.path.join(DATA_DIR, self.dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(
            save_dir,
            f'{data_split_name}_'
            f'transductive_val={self.args.transductive_val}_'
            f'val_test_ratio={self.args.val_ratio},{self.args.test_ratio}'
            f'_data_dists.pt'
        )

        if os.path.exists(save_path):
            print("Loading distance files from disk...")
            data.dists = torch.load(save_path, map_location=self.device)
        else:
            t1 = time.time()
            print("Precomputing distance files...")
            data.dists = torch.FloatTensor(precompute_dist_data(data.edge_index, data.num_nodes)).to(device)
            t2 = time.time()
            print(f"Finished precomputing distances on {data.dists.shape[1]} edge indices"
                  f" in {round(t2-t1, 2)}s. Saving to disk in {save_path}...")
            with open(save_path, 'wb') as f:
                torch.save(data.dists, f)

        new_data = AugmentedData(
            x=data.x,
            edge_index=data.edge_index,
            edge_labels=data.edge_labels,
            dists=data.dists,
        )
        return new_data

    def train(self):
        log_every = self.args.epoch_log if isinstance(self.args.epoch_log, int) else math.ceil(self.args.epoch_log * self.args.n_epochs)
        n_epochs = self.args.n_epochs

        log_every = log_every if isinstance(log_every, int) else int(n_epochs * log_every)
        t1 = time.time()
        running_loss = 0

        # Prepare data
        train_data = deal.prepare_data(
            'train',
            self.train_data,
        )
        val_data: AugmentedData = deal.prepare_data(
            'val',
            self.val_data,
        ).add_negative_samples(negative_sampling_ratio=self.args.negative_sampling_ratio_val)
        test_data: AugmentedData = self.test_data.add_negative_samples(
            negative_sampling_ratio=self.args.negative_sampling_ratio_test
        )
        t2 = time.time()

        InductiveDeal.show_dataset_info("Augmented Train data", train_data)
        InductiveDeal.show_dataset_info("Augmented Validation data", val_data)
        InductiveDeal.show_dataset_info("Augmented Test data", test_data)

        self.model = self.model.to(self.device)

        for epoch in range(n_epochs):
            # Shuffle training data and add negative samples

            batch_train_data: AugmentedData = train_data.add_negative_samples(
                negative_sampling_ratio=self.args.negative_sampling_ratio_train
            )

            train_edges, train_labels = batch_train_data.inputs

            # Calculate loss, backward and optimize
            self.optimizer.zero_grad()
            loss = self.model.default_loss(
                train_edges.t(),
                train_labels,
                batch_train_data,
                thetas=self.lambda_list
            )
            loss.backward()
            self.optimizer.step()

            # Calculate train/validation performance
            _loss = loss.item()
            running_loss += _loss
            if epoch % log_every == 0:
                avg_loss = running_loss / log_every
                # Transductive training performance
                train_scores = transductive_eval(
                    deal.model,
                    train_edges.t(),
                    train_labels,
                    train_data,
                    lambdas=self.lambda_list
                )

                val_data = val_data.shuffle()
                val_edges, val_labels = val_data.inputs

                if args.transductive_val:
                    # Transductive validation performance
                    val_scores = transductive_eval(
                        deal.model,
                        val_edges.t(),
                        val_labels,
                        val_data,
                        lambdas=self.lambda_list,
                    )
                else:
                    # Inductive validation performance\
                    val_scores = inductive_eval(
                        deal.model,
                        val_edges.t(),
                        val_labels,
                        val_data.x,
                        lambdas=self.inductive_lambda_list  # lambda_0 is 0 during inductive inference
                    )
                running_loss = 0.0
                print(
                    'Epoch: %s\n'
                    'Transductive Train:: [ROC-AUC: %.4f, Average Precision: %.4f, Train loss: %.4f, Average Train loss %.4f]\n'
                    '%s Validation [ROC-AUC: %.4f, Average Precision: %.4f]\n'
                    % (epoch + 1,
                       train_scores[0],
                       train_scores[1],
                       _loss,
                       avg_loss,
                       'Transductive' if self.args.transductive_val else 'Inductive',
                       val_scores[0], val_scores[1])
                )
        t3 = time.time()
        test_scores = inductive_eval(
            deal.model,
            test_data.edge_index.t(),
            test_data.edge_labels,
            test_data.x,
            lambdas=self.inductive_lambda_list  # lambda_0 is 0 during inductive inference
        )
        t4 = time.time()
        # Inductive test performance
        print('----------Inductive Test----------\n')
        print('\033[93m Total Load data time: %.2f s\033[0m' % (t2 - t1))
        print('\033[93m Total Train/val time: %.2f s\033[0m' % (t3 - t2))
        print('\033[93m Test time: %.2f s\033[0m' % (t4 - t3))
        print('\033[93m Total time: %.2f s\033[0m' % (t4 - t1))
        print(f'\033[93m ROC-AUC:{test_scores[0]:.4f} AP:{test_scores[1]:.4f}\033[0m')


if __name__ == '__main__':
    args = InductiveDeal.parse_args()
    deal = InductiveDeal(args)
    deal.train()
