"""
"""
import os
import json
import logging
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


class MIAModel(nn.module):

    def __init__(self, network, gcn_layer_configs=[], fcn_layer_configs=[],
                 optim_class="Adam", optim_args={},
                 scheduler_class=None, scheduler_args={},
                 cuda=True, devices=[0]):
        """
        Mutual Interactors Attention (MIA)

        args:
            network (PPINetwork) 
            layer_configs   ()
        """
        super().__init__()
        self.cuda = cuda
        self.device = devices[0]
        self.devices = devices
        self._build_optimizer(optim_class, optim_args, scheduler_class, scheduler_args)

        adj_matrix = torch.tensor(network.adj_matrix, dtype=torch.float)

        # build degree vector
        deg_vec = torch.sum(adj_matrix, axis=1, dtype=np.float)
        deg_vec = torch.power(deg_vec, -0.5)
        deg_vec = torch.tensor(deg_vec, dtype=torch.float)

        # convert adjacency to sparse matrix
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float)
        adj_rcnorm = torch.mul(torch.mul(deg_vec.view(1, -1), adj_matrix), 
                               deg_vec.view(-1, 1))
        adj_rnorm = torch.mul(D.view(1, -1), A)
        self.register_buffer("adj_rcnorm", adj_rcnorm)
        self.register_buffer("adj_rnorm", adj_rnorm)

        # gcn 
        self.gcn = nn.ModuleList()
        for layer_config in gcn_layer_configs: 
            layer = globals()[layer_config["class"]](**layer_config["args"])
            self.gcn.append(layer)
        
        # fcn
        self.fcn = nn.ModuleList()
        for layer_config in fcn_layer_configs: 
            layer = globals()[layer_config["class"]](**layer_config["args"])
            self.fcn.append(layer)
    
    def predict(self, inputs):
        """
        """
        return self.forward(inputs)
        
    def forward(self, inputs): 
        """
        """
        m, n = inputs.shape

        node_embeddings = torch.eye(self.adj_rcnorm.shape[0])
        for i, layer in enumerate(self.gcn):
            node_embeddings = layer(node_embeddings, self.adj_rcnorm)
        
        for i, layer in enumerate(self.fcn):
            node_embeddings = layer(node_embeddings)

        outputs = torch.matmul(inputs, self.adj_rcnorm)  # (m, n)
        outputs = torch.mul(outputs, node_embeddings)  # (d, m, n)
        outputs = torch.matmul(outputs, self.adj_rnorm)  # (d, m, n)

        return outputs

    def loss(self, outputs, targets):
        """
        Weighted binary cross entropy loss.
        """
        num_pos = 1.0 * targets.data.sum()
        num_neg = targets.data.nelement() - num_pos
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=num_neg / num_pos)
        return bce_loss(outputs, targets) 
    
    def score(self, dataloader, metric_configs=[], log_predictions=True):
        """
        """
        logging.info("Validation")
        self.eval()

        # move to cuda
        if self.cuda:
            self._to_gpu()

        metrics = Metrics(metric_configs)
        avg_loss = 0

        with tqdm(total=len(dataloader)) as t, torch.no_grad():
            for i, (inputs, targets, info) in enumerate(dataloader):
                # move to GPU if available
                if self.cuda:
                    inputs, targets = place_on_gpu([inputs, targets], self.device)

                # forward pass
                predictions = self.predict(inputs)

                labels = self._get_labels(targets)
                metrics.add(predictions, labels, info)

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
            self._to_gpu()

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
            for i, (inputs, targets, info) in enumerate(dataloader):
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
                    predictions = self.predict(inputs)
                    metrics.add(predictions, targets, info,
                                {"loss": loss})
                    del predictions

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
        # load optimizer
        if "params" in optim_args:
            for params_dict in optim_args["params"]:
                params_dict["params"] = self._modules[params_dict["params"]].parameters()
        else:
            optim_args["params"] = self.parameters()

        self.optimizer = getattr(optims, optim_class)(**optim_args)
        # load scheduler
        if scheduler_class is not None:
            self.scheduler = getattr(schedulers,
                                     scheduler_class)(self.optimizer,
                                                      **scheduler_args)
        else:
            self.scheduler = None
    
    def _to_gpu(self):
        """ Moves the model to the gpu. Should be reimplemented by child model for
        data parallel.
        """
        if self.cuda:
            self.to(self.device)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    Source: https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py  
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / torch.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
