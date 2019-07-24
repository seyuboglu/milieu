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

from milieu.methods.method import DPPMethod
from milieu.methods.gcn.gcn_model import GCNModel
from milieu.methods.lci.metrics import metrics
from milieu.methods.lci.utils import (save_checkpoint, load_checkpoint, 
                                      save_dict_to_json, RunningAverage)
    

class Milieu(nn.Module):

    def __init__(self, network, params):
        """
        Milieu model as described in {TODO}. 
        args:
            network (PPINetwork) 
            params   (dict) 
        """
        super(LCIModule, self).__init__()

        self.params = params 
        self.adj_matrix = network.adj_matrix 

        self._build_model()
    
    def _build_model(self):
        """
        Initializes the variables and parameters of the Milieu model. 
        See Methods, Equation (2) for corresponding mathematical definition. 
        """
        # degree vector, (D^{-0.5} in Equation (2))
        degree = np.sum(self.adj_matrix, axis=1, dtype=np.float)
        inv_sqrt_degree = np.power(degree, -0.5)
        inv_sqrt_degree = torch.tensor(inv_sqrt_degree, dtype=torch.float)

        # adjacency matrix of network, (A in Equation (2))
        adj_matrix = torch.tensor(self.dj_matrix, dtype=torch.float)

        # precompute the symmetric normalized adj matrix, used on the left of Equation (2)
        adj_matrix_left = torch.mul(torch.mul(inv_sqrt_degree.view(1, -1), 
                                              adj_matrix), 
                                    inv_sqrt_degree.view(-1, 1))

        # precompute the normalized adj matrix, used on the right of Equation (2)
        adj_matrix_right = torch.mul(inv_sqrt_degree.view(1, -1), 
                                     adj_matrix)
        self.register_buffer("adj_matrix_right", adj_matrix_right)
        self.register_buffer("adj_matrix_left", adj_matrix_left)
                
        # milieu weight vector, ('W' in Equation (2))
        self.milieu_weights = nn.Parameter(torch.ones(1, 1, adj_matrix.shape[0], 
                                           dtype=torch.float,
                                           requires_grad=True))

        # scaling parameter, ('a' in in Equation (2))
        self.scale = nn.Linear(1, 1)

        # the bias parameter, ('b' in Equation (2))
        self.bias = nn.Parameter(torch.ones(sizes=1, 
                                            dtype=torch.float,
                                            requires_grad=True))

    def forward(self, inputs):
        """
        Forward pass through the model. 
        Note: m is the # of diseases in the batch, n is the number of nodes 
        in the network
        """
        m, n = inputs.shape
        out = inputs  
        out = torch.matmul(inputs, self.A_left) 
        out = torch.mul(out, self.milieu_weights) 
        out = torch.matmul(out, self.A_right)  

        out = out.view(1, m * n).t()
        out = self.scaling(out) + self.bias
        out = out.view(m, n) 

        return out

    def predict(self, inputs):
        """
        """
        return self.forward(inputs)
        
    def loss(self, outputs, targets):
        """
        Weighted binary cross entropy loss.
        """
        # ignore -1 indices
        outputs = outputs[targets != -1]
        targets = targets[targets != -1]

        bce_loss = nn.BCEWithLogitsLoss()
        return bce_loss(outputs, targets) 
    
    def score(self, dataloader, metric_configs=[], log_predictions=True):
        """
        """
        logging.info("Validation")
        self.eval()

        # move to cuda
        if self.cuda:
            self.to(self.device)

        metrics = Metrics(metric_configs)
        avg_loss = 0

        with tqdm(total=len(dataloader)) as t, torch.no_grad():
            for i, (inputs, targets) in enumerate(dataloader):
                # move to GPU if available
                if self.cuda:
                    inputs, targets = place_on_gpu([inputs, targets], self.device)

                # forward pass
                predictions = self.predict(inputs)

                metrics.add(predictions, targets)

                # compute average loss and update the progress bar
                t.update()

        metrics.compute()
        return metrics
    
    def train_model(self):
        """
        Main training function.

        Trains the model, then collects metrics on the validation set. Saves
        weights on every epoch, denoting the best iteration by some specified
        metric.
        """
        logging.info(f'Starting training for {num_epochs} epoch(s)')

        # move to cuda
        if self.cuda:
            self.to(self.device)

        for epoch in range(self.params["num_epochs"]):
            logging.info(f'Epoch {epoch + 1} of {num_epochs}')

            # update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            train_metrics = self._train_epoch(dataloader, metric_configs,
                                              summary_period, writer)
            yield train_metrics

    def _train_epoch(self, dataloader,
                     metric_configs=[], summary_period=1, writer=None,
                     log_predictions=True):
        """ Train the model for one epoch
        Args:
            train_data  (DataLoader)
        """
        logging.info("Training")

        self.train()

        metrics = Metrics(metric_configs)

        avg_loss = 0

        with tqdm(total=len(dataloader)) as t:
            for i, (inputs, targets) in enumerate(dataloader):
                if self.cuda:
                    inputs, targets = place_on_gpu([inputs, targets], self.device)

                # forward pass
                outputs = self.forward(inputs)
                loss = self.loss(outputs, targets)

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss = loss.cpu().detach().numpy()
                # compute metrics periodically:
                if i % summary_period == 0:
                    metrics.add(outputs, targets, {"loss": loss})

                # compute average loss and update progress bar
                avg_loss = ((avg_loss * i) + loss) / (i + 1)
                if writer is not None:
                    writer.add_scalar(tag="loss", scalar_value=loss)
                t.set_postfix(loss='{:05.3f}'.format(float(avg_loss)))
                t.update()
                del loss, outputs, inputs, targets


        metrics.compute()
        return metrics
    
    def _build_optimizer(self,
                         optim_class="Adam", optim_args={},
                         scheduler_class=None, scheduler_args={}):
        """
        """
        self.optimizer = getattr(optims, optim_class)(self.parameters(), **optim_args)

        # load scheduler
        if scheduler_class is not None:
            self.scheduler = getattr(schedulers,
                                     scheduler_class)(self.optimizer,
                                                      **scheduler_args)
        else:
            self.scheduler = None


class MilieuDataset(Dataset):
    """
    """
    def __init__(self, diseases, network, frac_known=0.9):
        """
        """
        self.n = len(network)
        self.examples = [{"id": disease.id, 
                          "nodes": disease.to_node_array(network)}
                         for disease 
                         in diseases]
        self.frac_known = frac_known
    
    def __len__(self):
        """ 
        Returns the size of the dataset.
        """
        return len(self.examples)
    
    def get_ids(self):
        """
        Returns a set of all the disease ids in
        dataset.
        """
        return set([disease["id"] for disease in self.examples])

    def __getitem__(self, idx):
        nodes = self.examples[idx]["nodes"]
        np.random.shuffle(nodes)
        split = int(self.frac_known * len(nodes))

        known_nodes = nodes[:split]
        hidden_nodes = nodes[split:]

        X = torch.zeros(self.n, dtype=torch.float)
        X[known_nodes] = 1
        Y = torch.zeros(self.n, dtype=torch.float) 
        Y[hidden_nodes] = 1

        # ensure no data leakage
        assert(torch.dot(X, Y) == 0)

        return X, Y
