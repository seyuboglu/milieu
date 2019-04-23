"""
"""
from __future__ import division
import os
import json
import logging
from shutil import copyfile
from collections import defaultdict

import numpy as np
import networkx as nx
import torch 
import torch.nn as nn 
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix
import parse

from dpp.methods.method import DPPMethod
from dpp.methods.mia.mia_model import MIAModel
from dpp.methods.lci.metrics import metrics
from dpp.methods.lci.utils import (save_checkpoint, load_checkpoint, 
                                   save_dict_to_json, RunningAverage)


class MIA(DPPMethod):
    """ GCN method class
    """
    def __init__(self, network, diseases_dict, params):
        super().__init__(network, diseases_dict, params)

        self.dir = params["dir"]
        self.network = network 
        self.adjacency = self.network.adj_matrix
        self.params = params

        self.valid_dataset_args = params["valid_dataset_args"]
        self.train_dataset_args = params["train_dataset_args"]
        self.model_args = params["model_args"]
        self.train_args = params["train_args"]
        self.valid_args = params["valid_args"]
        
    def compute_scores(self, train_nodes, disease):
        """ Compute the scores predicted by GCN.
        Args: 
            
        """
        print(disease.id)
        model = MIAModel(self.network, **self.model_args)
        train_dataloader = DataLoader(TrainDataset(train_nodes, 
                                                   self.network,
                                                   **self.train_dataset_args),
                                      batch_size=self.params["batch_size"])
        
        valid_nodes = [node for node in disease.to_node_array(self.network) 
                       if node not in train_nodes]
        valid_dataloader = DataLoader(ValidDataset(train_nodes,
                                                   valid_nodes, 
                                                   self.network, 
                                                   **self.valid_dataset_args),
                                      batch_size=self.params["batch_size"])
        
        for epoch_num, train_metrics in enumerate(model.train_model(
                dataloader=train_dataloader, **self.train_args)):
            valid_metrics = model.score(valid_dataloader, **self.valid_args)

            logging.info(f"Train Metrics: {train_metrics.metrics}") 
            logging.info(f"Valid Metrics: {valid_metrics.metrics}")
        
        # get predictions 
        num_nodes = len(self.network)
        inputs = torch.zeros(1, num_nodes)
        inputs[0, train_nodes] = 1

        if model.cuda:
            inputs = inputs.cuda()

        outputs = model(inputs)
        scores = outputs.cpu().detach().numpy().squeeze()
        return scores


class TrainDataset(Dataset):

    def __init__(self, train_nodes, network, num_examples=200, frac_hidden=0.25):
        """
        """
        super()
        self.num_nodes = len(network)
        self.train_nodes = train_nodes
        self.num_examples = num_examples
        self.frac_hidden = frac_hidden

        adj_matrix = torch.tensor(network.adj_matrix, dtype=torch.float)
        deg_vec = torch.sum(adj_matrix, dim=1, dtype=torch.float)
        deg_vec = torch.pow(deg_vec, -0.5)
        self.adj_rcnorm = torch.mul(torch.mul(deg_vec.view(1, -1), adj_matrix), 
                               deg_vec.view(-1, 1)).to_sparse()
        self.adj_rnorm = torch.mul(deg_vec.view(1, -1), adj_matrix)

        self.normal = torch.distributions.normal.Normal(loc=0, scale=1)

    
    def __len__(self):
        """
        """
        return self.num_examples

    
    def __getitem__(self, idx):
        """
        """
        np.random.shuffle(self.train_nodes)
        split = int(self.frac_hidden * len(self.train_nodes))

        hidden_nodes = self.train_nodes[:split]
        known_nodes = self.train_nodes[split:]

        X = torch.zeros(self.num_nodes, dtype=torch.float)
        X[known_nodes] = 1

        mi_scores = torch.sparse.mm(self.adj_rcnorm, 
                                    torch.sum(self.adj_rnorm[known_nodes], 
                                              dim=0).unsqueeze(1)).squeeze()

        Y = -1 * torch.ones(self.num_nodes, dtype=torch.float) 
        for hidden_node in hidden_nodes:
            hidden_mi_score = mi_scores[hidden_node]

            normalized_scores = (mi_scores - hidden_mi_score) / torch.std(mi_scores)
            cdfs = self.normal.cdf(normalized_scores).numpy()
            probs = (cdfs / np.sum(cdfs))
            neg_node = np.random.choice(self.num_nodes, size=1, replace=False) #, p=probs)
            Y[neg_node] = 0
            Y[hidden_node] = 1

            # print(f"Hidden MI: {mi_scores[hidden_node]}, Neg MI: {mi_scores[neg_node]}")
            
        Y[known_nodes] = -1

        return X, Y


class ValidDataset(Dataset):

    def __init__(self, train_nodes, valid_nodes, network):
        """
        """
        self.num_nodes = len(network)
        self.train_nodes = train_nodes
        self.valid_nodes = valid_nodes
    
    def __len__(self):
        """
        """
        return 1

    def __getitem__(self, idx):
        """
        """

        X = torch.zeros(self.num_nodes, dtype=torch.float)
        X[self.train_nodes] = 1
        Y = torch.zeros(self.num_nodes, dtype=torch.float) 
        Y[self.valid_nodes] = 1

        # ensure no data leakage
        assert(torch.dot(X, Y) == 0)

        return X, Y