"""
"""
import os
import json
import logging
import math
from collections import defaultdict

import numpy as np
import networkx as nx
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import torch.optim as optims
import torch.optim.lr_scheduler as schedulers
from torch.nn import Linear
from tqdm import tqdm

from dpp.util import place_on_cpu, place_on_gpu
from dpp.methods.mia.metrics import Metrics


class GCNModel(nn.Module):

    def __init__(self, network, gcn_layer_configs=[], fcn_layer_configs=[],
                 dropout_prob=0.5, optim_class="Adam", optim_args={},
                 scheduler_class=None, scheduler_args={},
                 cuda=True, devices=[0]):
        """
        GCN (MIA)

        args:
            network (PPINetwork) 
            layer_configs   (list) 
        """
        super().__init__()
        self.cuda = cuda
        self.device = devices[0]
        self.devices = devices

        self.network = network

        adj_matrix = torch.tensor(network.adj_matrix, dtype=torch.float)

        # build degree vector
        deg_vec = torch.sum(adj_matrix, dim=1, dtype=torch.float)
        deg_vec = torch.pow(deg_vec, -0.5)

        # convert adjacency to sparse matrix
        adj_matrix_norm = torch.mul(torch.mul(deg_vec.view(1, -1), adj_matrix), 
                                    deg_vec.view(-1, 1)).to_sparse()
        self.register_buffer("adj_matrix_norm", adj_matrix_norm)
        if self.cuda:
            self.adj_matrix_norm = self.adj_matrix_norm.to(self.device)

        if "in_features" not in gcn_layer_configs[0]["args"]:
            self.features = False 
            gcn_layer_configs[0]["args"]["in_features"] = len(network)
        else:
            self.features = False

        # gcn 
        self.gcn = nn.ModuleList()
        for idx, layer_config in enumerate(gcn_layer_configs): 
            layer = globals()[layer_config["class"]](**layer_config["args"])
            self.gcn.append(layer)
        
        # fcn
        self.fcn = nn.ModuleList()
        for layer_config in fcn_layer_configs: 
            layer = globals()[layer_config["class"]](**layer_config["args"])
            self.fcn.append(layer)
                
        self._build_optimizer(optim_class, optim_args, scheduler_class, scheduler_args)

    def predict(self, inputs):
        """
        """
        return self.forward(inputs)
        
    def forward(self, inputs): 
        """
        """
        if not self.features:
            inputs = None

        outputs = inputs
        for i, layer in enumerate(self.gcn):
            outputs = layer(self.adj_matrix_norm, outputs)
                
        for i, layer in enumerate(self.fcn):
            outputs = layer(outputs)

        # add a batch size of 1, and remove trailing dimension
        outputs = outputs.squeeze().unsqueeze(0)
        return outputs

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
    
    def train_model(self, dataloader, num_epochs=20,
                    metric_configs=[], summary_period=1,
                    writer=None):
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

        for epoch in range(num_epochs):
            logging.info(f'Epoch {epoch + 1} of {num_epochs}')

            # update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                learning_rate = self.scheduler.get_lr()[0]
                logging.info(f"- Current learning rate: {learning_rate}")

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


class GraphConvolutionLayer(nn.Module):
    """
    GCN layer 
    Similar to: https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py  
    """

    def __init__(self, in_features, out_features, bias=True, 
                 activation="relu"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, inputs=None):
        # if inputs is None:
        #     neighborhood = adj.to_dense()
        #     if self.self_dropout is not None and self.training:
        #         mask = (torch.rand(self.weight.shape[0]//2) > 
        #                 self.self_dropout).type(torch.float).to(self.weight.device)
        #         self_weight = self.weight[:self.weight.shape[0]//2] * mask.unsqueeze(-1)
        #     else:
        #         self_weight = self.weight[:self.weight.shape[0]//2]

        #     outputs = torch.mm(neighborhood, self.weight[self.weight.shape[0]//2:])
        #     outputs += self_weight
        # else:
        #     neighborhood = torch.spmm(adj, inputs) 
        #     if self.self_dropout is not None and self.training:
        #         mask = (torch.rand(inputs.shape[0]) > 
        #                 self.self_dropout).type(torch.float).to(inputs.device)
        #         inputs = inputs * mask.unsqueeze(-1)
        #     outputs = torch.mm(torch.cat([inputs, neighborhood], dim=1), self.weight)

        if inputs is None:
            neighborhood = adj
        else:
            neighborhood = torch.spmm(adj, inputs)
        outputs = torch.mm(neighborhood, self.weight)



        if self.bias is not None:
            outputs = outputs + self.bias
        
        if self.activation == "relu":
            return nn.functional.relu(outputs)
        elif self.activation is None:
            return outputs
        else:
            raise ValueError(f"Activation {self.activation} not supported.")

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class FullyConnectedLayer(nn.Module):
    """
    Simple fully connected layer. 
    """

    def __init__(self, in_features, out_features, bias=True, 
                 dropout=0.5, activation=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.activation = activation

        self.linear_layer = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, inputs):
        outputs = self.linear_layer(inputs)
        
        if self.dropout is not None:
            self.dropout(outputs)
        
        if self.activation == "relu":
            return nn.functional.relu(outputs)
        elif self.activation is None:
            return outputs
        else:
            raise ValueError(f"Activation {self.activation} not supported.")
