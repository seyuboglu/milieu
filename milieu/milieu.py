#!/usr/bin/env python
"""
Implementation of Milieu model as described in {TODO}.

Includes:
Milieu  a torch.nn.Module that implements a trainable Milieu model.
MilieuDataset   a torch.util.data.Dataset that serves disease-protein associations
"""
import os
import json
import logging
from shutil import copyfile
from collections import defaultdict
from copy import deepcopy

import numpy as np
import networkx as nx
import torch 
import torch.nn as nn 
from torch.autograd import Variable
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix
import parse

from milieu.methods.method import DPPMethod
from milieu.data.associations import load_diseases
from milieu.metrics import compute_metrics
from milieu.util import set_logger, load_mapping

__author__ = "Evan Sabri Eyuboglu"
    
class Milieu(nn.Module):
    """
    Milieu model as described in {TODO}. 
    Milieu training is parameterized by the self.params dictionary. The default dictionary
    is updated with the params passed to __init__. 
    """
    default_params = {
        "cuda": True,
        "device": 0,
            
        "batch_size": 200,
        "num_workers": 4,
            
        "optim_class": "Adam",
        "optim_args": {
            "lr": 1e-1,
            "weight_decay": 0
        }, 

        "metric_configs": [
            {
                "name": "recall_at_25",
                "fn": "batch_recall_at", 
                "args": {"k":25}
            }
        ]
    }

    def __init__(self, network, params):
        """
        Milieu model as described in {TODO}. 
        args:
            network (PPINetwork) The PPINetwork to use. 
            params   (dict) Params to update in default_params. 
        """
        super().__init__()

        set_logger()
        logging.info("Milieu")

        # override default parameters
        logging.info("Setting parameters...")
        self.params = deepcopy(self.default_params)
        self.params.update(params)

        self.network = network
        self.adj_matrix = network.adj_matrix 

        self.genbank_to_entrez = load_mapping("data/protein/genbank_to_entrez.txt",
                                              b_transform=int, delimiter='\t')
        self.entrez_to_genbank = {entrez: genbank 
                                  for genbank, entrez in self.genbank_to_entrez.items()}

        logging.info("Building model...")
        self._build_model()
        logging.info("Building optimizer...")
        self._build_optimizer()
        logging.info("Done.")
    
    def _build_model(self):
        """
        Initialize the variables and parameters of the Milieu model. 
        See Methods, Equation (2) for corresponding mathematical definition. 
        """
        # degree vector, (D^{-0.5} in Equation (2))
        degree = np.sum(self.adj_matrix, axis=1, dtype=np.float)
        inv_sqrt_degree = np.power(degree, -0.5)
        inv_sqrt_degree = torch.tensor(inv_sqrt_degree, dtype=torch.float)

        # adjacency matrix of network, (A in Equation (2))
        adj_matrix = torch.tensor(self.adj_matrix, dtype=torch.float)

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
        self.bias = nn.Parameter(torch.ones(size=(1,), 
                                            dtype=torch.float,
                                            requires_grad=True))

    def forward(self, inputs):
        """
        Forward pass through the model. See Methods, Equation (2).
        args:
            inputs (torch.Tensor) an (m, n) binary torch tensor where m = # of diseases
            in batch and n = # of proteins in the PPI network
        returns:
            out (torch.Tensor) an (m, n) torch tensor. Element (i, j) is the activations 
            for disease i and protein j. Note: these are activations, not probabilities.
            Use torch.sigmoid to convert to probabilties. 
        """
        m, n = inputs.shape
        out = inputs  
        out = torch.matmul(inputs, self.adj_matrix_left) 
        out = torch.mul(out, self.milieu_weights) 
        out = torch.matmul(out, self.adj_matrix_right)  

        out = out.view(1, m * n).t()
        out = self.scale(out) + self.bias
        out = out.view(m, n) 

        return out

    def predict(self, inputs):
        """
        Make probabilistic predictions for novel protein associaitions for a batch of 
        diseases.
        args:
            inputs (torch.Tensor) an (m, n) binary torch tensor where m = # of diseases
            in batch and n = # of proteins in the PPI network
        returns:
            out (torch.Tensor) an (m, n) torch tensor. Element (i, j) is the probability 
            protein j is associated with disease i and protein j.
        """
        return torch.sigmoid(self.forward(inputs))
    
    def discover(self, entrez_ids=None, genbank_ids=None, top_k=10):
        """
        Get the top k proteins with the highest probability of association for a 
        phenotype of interest. Must provide proteins known to be associated with the 
        phenotype via either the entrez_ids or genbank_ids argument. 
        args:
            entrez_ids (iterable) a set of entrez ids representing proteins known to be 
            associated with a phenotype of interest. Note: must provide either entrez_ids
            or genbank ids, but not both. 

            genbank_ids (iterable) a set of genbank ids representing proteins known to be
            associated with a phenotype of interest. Note: must provide either entrez_ids 
            or genbank ids, but not both. 

            top_k (int) the number of predicted proteins to return
        returns:
            top_k_entrez/genbank    (list(tuple)) returns a list of tuples of the form
            (entrez/genbank_id, probability of association). Includes top k predictions
            with highest probability. 
        """
        if (entrez_ids is not None) == (genbank_ids is not None):
            raise ValueError("Must provide either Entrez ids or GenBank ids, not both.")
        
        if genbank_ids is not None:
            # get entrez from genbank ids
            entrez_ids = [self.genbank_to_entrez[genbank] 
                          for genbank in genbank_ids if genbank in self.genbank_to_entrez]
            missing_ids = [genbank for genbank in genbank_ids 
                           if genbank not in self.genbank_to_entrez]
            if missing_ids:
                logging.warning(f"Could not find entrez_ids for {missing_ids}")

        # build model input vector 
        input_nodes = self.network.get_nodes(entrez_ids)
        inputs = torch.zeros((1, len(self.network)), dtype=torch.float)
        inputs[0, input_nodes] = 1
        
        if self.params["cuda"]:
            inputs = inputs.to(self.params["device"])

        probs = self.predict(inputs).cpu().detach().numpy().squeeze()

        # get top k predictions
        ranking = np.arange(len(self.network))[np.argsort(-probs)]
        top_k_nodes = []
        for node in ranking:
            if node not in input_nodes:
                top_k_nodes.append(node)
                if len(top_k_nodes) == top_k:
                    break

        top_k_probs = probs[top_k_nodes]
        top_k_entrez = list(self.network.get_proteins(top_k_nodes))

        if genbank_ids is not None:
            top_k_genbank = [self.entrez_to_genbank[entrez] 
                             for entrez in top_k_entrez 
                             if entrez in self.entrez_to_genbank]
            return list(zip(top_k_genbank, top_k_probs))

        return list(zip(top_k_entrez, top_k_probs))

    def loss(self, outputs, targets):
        """
        Compute weighted, binary cross-entropy loss as described in Methods, Equation (3).
        Positive examples are weighted {# of negative examples} / {# of positive examples}
        and negative examples are weighted 1. 
        args:
            outputs (torch.Tensor) An (m, n) torch tensor. Element (i, j) is the 
            activation for disease i and protein j. Note: these are activations, not 
            probabilities. We use BCEWithLogitsLoss which combines the sigmoid with 
            the loss for numerical stability. 

            targets (torch.Tensor) An (m, n) binary tensor. Element (i, j) indicates
            whether protein j is in the held-out set of proteins associated with disease
            i. 

        returns:
            out (torch.Tensor) A scalar loss.
        """
        num_pos = 1.0 * targets.data.sum()
        num_neg = targets.data.nelement() - num_pos
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=num_neg / num_pos)
        return bce_loss(outputs, targets) 
    
    def train_model(self, train_dataset, valid_dataset=None): 
        """
        Train the Milieu model on train_dataset. Parameters for training including
        "num_epochs" and "optimizer_class" should be specified in the params dict 
        passed in at __init__.  Optionally validate the model on a validation dataset
        on each epoch. Computes metrics specified in params["metric_configs"] on each
        epoch. 
        args:
            train_dataset (MilieuDataset) A milieu dataset of training diseases
            valid_dataset (MilieuDataset) A milieu dataset of validatio diseases
        returns:
            train_metrics   (list(dict)) train_metrics[i] is a dictionary mapping metric
            names to their values on epoch i
            valid_metrics   (list(dict)) like train_metrics but for validation metrics 
        """
        logging.info(f'Starting training for {self.params["num_epochs"]} epoch(s)')

        # move to cuda
        if self.params["cuda"]:
            self.to(self.params["device"])

        train_metrics = []
        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=self.params["batch_size"], 
                                      shuffle=True,
                                      num_workers=self.params["num_workers"],
                                      pin_memory=self.params["device"])
        
        validate = valid_dataset is not None
        if validate:
            valid_metrics = []
            valid_dataloader = DataLoader(valid_dataset,
                                          batch_size=self.params["batch_size"], 
                                          shuffle=True,
                                          num_workers=self.params["num_workers"],
                                          pin_memory=self.params["device"])
        
        all_train_metrics = []
        for epoch in range(self.params["num_epochs"]):
            logging.info(f'Epoch {epoch + 1} of {self.params["num_epochs"]}')

            metrics = self._train_epoch(train_dataloader)
            train_metrics.append(metrics)

            if validate:
                metrics = self.score(valid_dataloader)
                valid_metrics.append(metrics)

        return train_metrics, valid_metrics if validate else train_metrics

    def _train_epoch(self, dataloader, metric_configs=[]):
        """ Train the model for one epoch
        args:
            train_data  (DataLoader) A dataloader wrapping a MilieuDataset
            metric_configs  (list(dict)) A list of metric configuration dictionary. Each
            config dict should include "name", "fn", and "args". "fn" should be the name
            of a function in milieu.metrics. See default params for an example. 
        return:
            metrics (dict)  Dictionary mapping metric "name" to value. 
        """
        logging.info("Training")

        self.train()

        metrics = defaultdict(list)

        avg_loss = 0

        with tqdm(total=len(dataloader)) as t:
            for i, (inputs, targets) in enumerate(dataloader):
                if self.params["cuda"]:
                    inputs = inputs.to(self.params["device"])
                    targets = targets.to(self.params["device"])

                # forward pass
                outputs = self.forward(inputs)
                loss = self.loss(outputs, targets)

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss = loss.cpu().detach().numpy()
                # compute metrics
                probs = torch.sigmoid(outputs)
                compute_metrics(probs.cpu().detach().numpy(),
                                targets.cpu().detach().numpy(),
                                metrics,
                                self.params["metric_configs"])

                # compute average loss and update progress bar
                avg_loss = ((avg_loss * i) + loss) / (i + 1)
          
                t.set_postfix(loss='{:05.3f}'.format(float(avg_loss)))
                t.update()
                del loss, outputs, inputs, targets

        metrics = {name: np.mean(values) for name, values in metrics.items()}
        return metrics
    
    def score(self, dataloader, metric_configs=[]):
        """ Evaluate the model on the data in dataloader and the metrics in 
            metric_configs.
        args:
            train_data  (DataLoader) A dataloader wrapping a MilieuDataset
            metric_configs  (list(dict)) A list of metric configuration dictionary. Each
            config dict should include "name", "fn", and "args". See default params for
            example. 
        return:
            metrics (dict)  Dictionary mapping metric "name" to value. 
        """
        logging.info("Validation")
        self.eval()

        # move to cuda
        if self.params["cuda"]:
            self.to(self.params["device"])

        metrics = defaultdict(list)
        avg_loss = 0

        with tqdm(total=len(dataloader)) as t, torch.no_grad():
            for i, (inputs, targets) in enumerate(dataloader):
                # move to GPU if available
                if self.params["cuda"]:
                    inputs = inputs.to(self.params["device"])
                    targets = targets.to(self.params["device"])

                # forward pass
                probs = self.predict(inputs)
                compute_metrics(probs.cpu().detach().numpy(),
                                targets.cpu().detach().numpy(),
                                metrics,
                                self.params["metric_configs"])

                # compute average loss and update the progress bar
                t.update()

        return metrics
    
    def _build_optimizer(self,
                         optim_class="Adam", optim_args={}):
        """
        Build the optimizer. 
        args:
            optim_class (str) The name of an optimizer class from torch.optim
            opteim_args (args) The args for the optimizer
        """
        optim_class = getattr(optim, self.params["optim_class"])
        self.optimizer = optim_class(self.parameters(), **self.params["optim_args"])
    
    def save_weights(self, destination):
        """
        Save the model weights. 
        args:
            destination (str)   path where to save weights
        """
        torch.save(self.state_dict(), destination)

    def load_weights(self, src_path):
        """
        Load model weights. 
        args:
            src_path (str) path to the weights files.
            substitution_res (list(tuple(str, str))) list of tuples like
                    (regex_pattern, replacement). re.sub is called on each key in the dict
        """
        if self.params["cuda"]:
            src_state_dict = torch.load(src_path, 
                                        map_location=torch.device(self.params["device"]))
        else:
            src_state_dict = torch.load(src_path)

        self.load_state_dict(src_state_dict, strict=False)
        n_loaded_params = len(set(self.state_dict().keys()) & set(src_state_dict.keys()))
        n_tot_params = len(src_state_dict.keys())
        if n_loaded_params < n_tot_params:
            logging.info("Could not load these parameters due to name mismatch: " +
                         f"{set(src_state_dict.keys()) - set(self.state_dict().keys())}")
        logging.info(f"Loaded {n_loaded_params}/{n_tot_params} pretrained parameters " +
                     f"from {src_path}.")


class MilieuDataset(Dataset):

    def __init__(self, network, diseases=None, diseases_path=None, frac_known=0.9):
        """
        PyTorch dataset that holds disease-protein association sets and serves them to the 
        Milieu for training. During training we simulate disease protein discovery by 
        splitting each association set into an input set and a target set. Each time we
        access an association set from this dataset we randomly sample 90% of associations 
        for the input set and use the remaining 10% for the target set. See Methods. 
        args:
            network (PPINetwork)    PPINetwork being used by the Milieu model
            diseases    (list(Disease)) list of milieu.data.associations.Disease. Note:
            must provide either diseases or diseases_path but not both. 
            diseases_path (str) a path to a csv file of disease associations readable by 
            load_diseases. Note: must provide either diseases or diseases_path.
            frac_known  (float)   fraction of each association set used for input set and
            target set. 
        """
        if(diseases is not None) == (diseases_path is not None):
            raise ValueError("Must provide either a list of Diseases or a path to " + 
                             "a *.csv of disease associations readable by load_diseases.")
        
        if diseases_path is not None:
            diseases = list(load_diseases(diseases_path).values())

        self.n = len(network)
        self.examples = [{"id": disease.id, 
                          "nodes": disease.to_node_array(network)}
                         for disease 
                         in diseases]
        self.frac_known = frac_known
    
    def __len__(self):
        """ Returns the size of the dataset."""
        return len(self.examples)
    
    def get_ids(self):
        """ Get the set of all the disease ids in the dataset."""
        return set([disease["id"] for disease in self.examples])

    def __getitem__(self, idx):
        """
        Get an association split into an input and target set as described in Methods. 
        args:
            idx (int) The index of the association set in the dataset. 
        returns:
            inputs (torch.Tensor) an (n,) binary torch tensor indicating the proteins in
            the input set.
            targets (torch.Tensor) an (n,) binary torch tensor indicating the proteins in
            the target set.
        """
        nodes = self.examples[idx]["nodes"]
        np.random.shuffle(nodes)
        split = int(self.frac_known * len(nodes))

        known_nodes = nodes[:split]
        hidden_nodes = nodes[split:]

        inputs = torch.zeros(self.n, dtype=torch.float)
        inputs[known_nodes] = 1
        targets = torch.zeros(self.n, dtype=torch.float) 
        targets[hidden_nodes] = 1

        # ensure no data leakage
        assert(torch.dot(inputs, targets) == 0)

        return inputs, targets
