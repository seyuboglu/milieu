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
from dpp.methods.gcn.gcn_model import GCNModel
from dpp.methods.lci.metrics import metrics
from dpp.methods.lci.utils import (save_checkpoint, load_checkpoint, 
                                   save_dict_to_json, RunningAverage)


class GCN(DPPMethod):
    """ GCN method class
    """
    def __init__(self, network, diseases_dict, params):
        super().__init__(network, diseases_dict, params)

        self.dir = params["dir"]
        self.network = network 
        self.adjacency = self.network.adj_matrix
        self.params = params

        self.dataset_args = params["dataset_args"]
        self.model_args = params["model_args"]
        self.train_args = params["train_args"]
        self.valid_args = params["valid_args"]
        
    def compute_scores(self, train_nodes, disease):
        """ Compute the scores predicted by GCN.
        Args: 
            
        """
        model = GCNModel(self.network, **self.model_args)

        train_dataloader = DataLoader(GCNDataset(train_nodes, 
                                                 self.network,
                                                 split="train",
                                                 **self.dataset_args), batch_size=1)
        
        valid_nodes = [node for node in disease.to_node_array(self.network) 
                       if node not in train_nodes]
        valid_dataloader = DataLoader(GCNDataset(valid_nodes,
                                                 self.network, 
                                                 split="valid",
                                                 **self.dataset_args), batch_size=1)
        
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


class GCNDataset(Dataset):

    def __init__(self, pos_nodes, network, split="train", num_examples=200):
        """
        """
        self.num_nodes = len(network)
        self.pos_nodes = pos_nodes
        self.num_examples = num_examples
        self.split = split

    def __len__(self):
        """
        """
        return self.num_examples

    def __getitem__(self, idx):
        """
        """
        if self.split == "train":
            Y = -1 * torch.ones(self.num_nodes, dtype=torch.float) 
            neg_nodes = torch.randint(high=self.num_nodes, size=(len(self.pos_nodes),))
            Y[neg_nodes] = 0
        elif self.split == "test" or self.split == "valid":
            Y = torch.zeros(self.num_nodes, dtype=torch.float) 
        Y[self.pos_nodes] = 1
            
        return [], Y
