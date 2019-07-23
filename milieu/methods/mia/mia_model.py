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

from milieu.util import place_on_cpu, place_on_gpu
from milieu.methods.mia.metrics import Metrics
from milieu.data.embeddings import load_embeddings


class MIAModel(nn.Module):

    def __init__(self, network, embeddings_path,
                 dropout_prob=0.5, optim_class="Adam", optim_args={},
                 scheduler_class=None, scheduler_args={},
                 cuda=True, devices=[0]):
        """
        Mutual Interactors Attention (MIA)

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
        self.register_buffer("adj_matrix", adj_matrix)

        # build degree vector
        deg_vec = torch.sum(adj_matrix, dim=1, dtype=torch.float)
        deg_vec = torch.pow(deg_vec, -0.5)

        # convert adjacency to sparse matrix
        adj_rcnorm = torch.mul(torch.mul(deg_vec.view(1, -1), adj_matrix), 
                                         deg_vec.view(-1, 1)).to_sparse()
        adj_rnorm = torch.mul(deg_vec.view(1, -1), adj_matrix).to_sparse()
        self.register_buffer("adj_rcnorm", adj_rcnorm)
        self.register_buffer("adj_rnorm", adj_rnorm)

        embeddings = torch.tensor(load_embeddings(embeddings_path, network), 
                                       dtype=torch.float)
        self.register_buffer("embeddings", embeddings)

        self.att_proj_layer = nn.Linear(in_features=self.embeddings.shape[1], 
                                        out_features=self.embeddings.shape[1], bias=True)

        self.weight_layer = nn.Linear(in_features=self.embeddings.shape[1] * 2, 
                                      out_features=1, bias=False)
        self.final_layer = nn.Linear(in_features=1, out_features=1, bias=False)
        
        self.dropout = nn.Dropout(p=dropout_prob)
        self._build_optimizer(optim_class, optim_args, scheduler_class, scheduler_args)

    def predict(self, inputs):
        """
        """
        return self.forward(inputs)
        
    def forward(self, inputs): 
        """
        """
        batch_size, num_nodes = inputs.shape
        assert(batch_size == 1)
        inputs = inputs.squeeze(0)

        # currently only works with batch_size 1
        pos_embeddings = self.embeddings[torch.nonzero(inputs), :].squeeze(1)  # num_pos x d
        pos_embeddings_proj = self.att_proj_layer(pos_embeddings)  # num_pos x d
        att_scores = torch.matmul(pos_embeddings_proj, self.embeddings.t())  # num_pos x num_nodes
        pos_adj = self.adj_matrix[torch.nonzero(inputs), :].squeeze(1)
        att_scores = pos_adj * att_scores  # num_pos x num_nodes
        att_probs = torch.softmax(att_scores, dim=0)   # num_pos x num_nodes
        context_embeddings = torch.matmul(att_probs.t(), pos_embeddings) # num_nodes x d
        context_embeddings = torch.cat((self.embeddings, context_embeddings), dim=1)
        mutual_node_weights = torch.sigmoid(self.weight_layer(context_embeddings))
        
        inputs = inputs.unsqueeze(1)
        outputs = torch.sparse.mm(self.adj_rcnorm.t(), inputs).t()  # (m, n)
        outputs = torch.mul(outputs, mutual_node_weights.t())  # (m, n)
        outputs = outputs.squeeze(0).unsqueeze(1)
        outputs = torch.sparse.mm(self.adj_rnorm.t(), outputs).t()  # (m, n)
        outputs = self.final_layer(outputs.unsqueeze(-1)).squeeze(-1)
        print(self.final_layer.weight)

        return outputs

    def loss(self, outputs, targets):
        """
        Weighted binary cross entropy loss.
        """
        # ignore -1 indices
        outputs = outputs[targets != -1]
        targets = targets[targets != -1]
        print(targets)
        print(outputs)
        print(torch.sigmoid(outputs))

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
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
