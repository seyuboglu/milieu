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
from dpp.methods.lci.metrics import metrics
from dpp.methods.lci.utils import (save_checkpoint, load_checkpoint, 
                                   save_dict_to_json, RunningAverage)


class LCI(DPPMethod):
    """ GCN method class
    """
    def __init__(self, network, diseases, params):
        super().__init__(network, diseases, params)

        self.dir = params["dir"]
        self.adjacency = self.network.adj_matrix
        self.diseases = diseases
        self.params = params
        print(self.params)
        if "load_dir" in self.params:
            self.load_method()
        else:
            self.train_method(diseases)
        
        self.curr_fold = None
    
    def load_method(self):
        """
        """
        logging.info("Loading Params...")
        with open(os.path.join(self.params["load_dir"], "params.json")) as f:
            params = json.load(f)["process_params"]["method_params"]
        params.update(self.params)
        self.params = params
        
        logging.info("Loading Models...")
        self.folds_to_models = {}
        for model_file in os.listdir(os.path.join(self.params["load_dir"], "models")):
            split = parse.parse("model_{}.tar", model_file)[0]
            self.folds_to_models[split] = os.path.join(self.params["load_dir"], 
                                                       "models", 
                                                       model_file)


    def train_method(self, diseases):
        """
        """
        logging.info("Training Models...")
        folds_to_diseases = defaultdict(set)
        for disease in diseases.values():
            if disease.split == "none":
                continue
            folds_to_diseases[disease.split].add(disease)
        
        self.folds_to_models = {}
        if not(os.path.exists(os.path.join(self.dir, "models"))):
            os.mkdir(os.path.join(self.dir, "models"))
        
        for test_fold in folds_to_diseases.keys(): 
            logging.info("Training model for test {}".format(test_fold))
            val_fold = str((int(test_fold) - 1) % len(folds_to_diseases))
            test_dataset = DiseaseDataset([disease 
                                           for disease in folds_to_diseases[test_fold]],
                                           self.network)
            val_dataset = DiseaseDataset([disease 
                                          for disease in folds_to_diseases[val_fold]],
                                          self.network) 
            train_dataset = DiseaseDataset([disease  
                                             for fold, diseases in folds_to_diseases.items()
                                             if fold != test_fold and fold != val_fold
                                             for disease in diseases], 
                                            self.network)
            
            # ensure no data leakage
            assert(not set.intersection(*[test_dataset.get_ids(), 
                                          train_dataset.get_ids()]))
            assert(not set.intersection(*[val_dataset.get_ids(),
                                          train_dataset.get_ids()]))

            model = self.train_model(train_dataset, val_dataset)
            path = os.path.join(self.dir, "models/model_{}.tar".format(test_fold))
            torch.save(model.state_dict(), path)
            self.folds_to_models[test_fold] = path

    
    def train_model(self, train_dataset, val_dataset):
        """ Trains the underlying model
        """
        train_dl = DataLoader(train_dataset, 
                              batch_size=self.params["batch_size"], 
                              shuffle=True,
                              num_workers=self.params["num_workers"],
                              pin_memory=self.params["cuda"])
    
        dev_dl = DataLoader(val_dataset, 
                            batch_size=self.params["batch_size"], 
                            shuffle=True,
                            num_workers=self.params["num_workers"],
                            pin_memory=self.params["cuda"])


        model = LCIModule(self.params, self.adjacency)

        if self.params["cuda"]:
            model = model.cuda()
        optimizer = Adam(model.parameters(), lr=self.params["learning_rate"], 
                         weight_decay=self.params["weight_decay"])
        
        logging.info("Starting training for {} epoch(s)".format(self.params["num_epochs"]))
        model.train()
        train_and_evaluate(
            model,
            train_dl,
            dev_dl,
            optimizer,
            bce_loss,
            metrics,
            self.params,
            self.dir
        )
        model.eval()
        return model.cpu()


    def compute_scores(self, train_pos, disease):
        """ Compute the scores predicted by GCN.
        Args: 
            
        """
        val_pos = None
        # Adjacency: Get sparse representation of ppi_adj
        N, _ = self.adjacency.shape
        X = torch.zeros(1, N)
        X[0, train_pos] = 1
        if self.params["cuda"]:
            X = X.cuda()
            
        if disease.split != self.curr_fold:
            model = LCIModule(self.params, self.adjacency)
            model.load_state_dict(torch.load(self.folds_to_models[disease.split]))          
            model.eval()
            model.cuda()  
            self.curr_model = model 
            self.curr_fold = disease.split
        Y = self.curr_model(X)
        scores = Y.cpu().detach().numpy().squeeze()
        return scores


class LCIModule(nn.Module):

    def __init__(self, params, adjacency):
        super(LCIModule, self).__init__()
        # build degree vector
        D = np.sum(adjacency, axis=1, dtype=np.float)
        D = np.power(D, -0.5)
        D = torch.tensor(D, dtype=torch.float)

        # convert adjacency to sparse matrix
        A = torch.tensor(adjacency, dtype=torch.float)
        A_left = torch.mul(torch.mul(D.view(1, -1), A), D.view(-1, 1))
        A_right = torch.mul(D.view(1, -1), A)
        self.register_buffer("A_right", A_right)
        self.register_buffer("A_left", A_left)
        
        N, _ = A.shape
        self.d = params["d"]

        # coo = coo_matrix(A)
        # i = torch.LongTensor(np.vstack((coo.row, coo.col)))
        # v = torch.FloatTensor(coo.data)
        # A_sparse = torch.sparse.FloatTensor(i, v, torch.Size(coo.shape))
        # self.register_buffer("A_sparse", A_sparse)
        
        if params["initialization"] == "ones":
            self.E = nn.Parameter(torch.ones(self.d, 1, N, 
                                             dtype=torch.float,
                                             requires_grad=True))
        elif params["initialization"] == "zeros":
            self.E = nn.Parameter(torch.zeros(self.d, 1, N, 
                                              dtype=torch.float,
                                              requires_grad=True))
        else:
            logging.error("Initialization not recognized.")
        self.relu = nn.ReLU()

        units = getattr(params, "linear_layer_units", [])

        self.b = nn.Parameter(torch.ones(1, 
                                         dtype=torch.float,
                                         requires_grad=True))
        self.linear_layer = nn.Linear(self.d, 1)

    def forward(self, input, test=False):
        """
        Forward pass through the model. 
        Note: m is the # of diseases in the batch, n is the number of nodes 
        in the network, d is the depth of the embedding. 
        """
        m, n = input.shape
        X = input  # (m, n)
        X = torch.matmul(X, self.A_left)  # (m, n)
        X = torch.mul(X, self.E)  # (d, m, n)
        X = torch.matmul(X, self.A_right)  # (d, m, n)

        X = X.view(self.d, m * n).t()
        X = self.linear_layer(X) + self.b
        X = X.view(m, n)  # (m, n)
        return X


class DiseaseDataset(Dataset):
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


def fetch_dataloader(diseases, protein_to_node, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the database
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}
    datasets = {} 

    dataset = DiseaseDataset(diseases, protein_to_node) 

    # ensure no data leakage
    assert(not set.intersection(*[dataset.get_ids() for dataset in datasets.values()]))

    dataloader = DataLoader(dataset, 
                            batch_size=params["batch_size"], 
                            shuffle=True,
                            num_workers=params["num_workers"],
                            pin_memory=params["cuda"])

    return dataloader


def bce_loss(outputs, labels, params):
    """
    """
    num_pos = 1.0 * labels.data.sum()
    num_neg = labels.data.nelement() - num_pos
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=num_neg / num_pos)
    return bce_loss(outputs, labels)


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and 
                computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches
                 training data
        metrics: (dict) a dictionary of functions that compute a metric using the output 
                and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    loss_avg = RunningAverage()

    outputs_labels = []
    losses = []

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (data_batch, labels_batch) in enumerate(dataloader):
            # ensure no data leakage
            assert(torch.dot(data_batch.view(-1), labels_batch.view(-1)) == 0)

            # move to GPU if available
            if params["cuda"]:
                labels_batch = labels_batch.cuda(params["cuda_gpu"])
                data_batch = data_batch.cuda(params["cuda_gpu"])

            # compute model output and loss
            if params["cuda"]:
                model.cuda(params["cuda_gpu"])
            
            output_batch = model(data_batch)
            loss = loss_fn(output_batch, labels_batch, getattr(params, 'loss_params', None))

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params["save_summary_steps"] == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                if(type(output_batch) is tuple):
                    # output_batch is first element in tuple 
                    output_batch = output_batch[0]
                outputs_labels.append((output_batch.data.cpu(), labels_batch.data.cpu()))

            # update the average loss
            losses.append(loss.data.cpu().numpy())
            loss_avg.update(loss.data)

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    outputs, labels = zip(*outputs_labels)
    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)
    summary = {metric : metrics[metric](outputs, labels)
               for metric in metrics}
    summary['loss'] = np.mean(losses)

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in summary.items())
    logging.info("- Train metrics: " + metrics_string)


def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    outputs_labels = []
    losses = []

    # compute metrics over the dataset
    with tqdm(total=len(dataloader)) as t:
        for data_batch, labels_batch in dataloader:
            # ensure no data leakage
            assert(torch.dot(data_batch.view(-1), labels_batch.view(-1)) == 0)
            # move to GPU if available
            if params["cuda"]:
                labels_batch = labels_batch.cuda(params["cuda_gpu"])
                data_batch = data_batch.cuda(params["cuda_gpu"])

            output_batch = model(data_batch)
            if params["cuda"]:
                model.cuda(params["cuda_gpu"])

            loss = loss_fn(output_batch, labels_batch, getattr(params, 'loss_params', None))
            losses.append(loss.data.cpu().numpy())


            # extract data from torch Variable, move to cpu, convert to numpy arrays
            if(type(output_batch) is tuple):
                # output_batch is first element in tuple 
                output_batch = output_batch[0]
            outputs_labels.append((output_batch.data.cpu(), labels_batch.data.cpu()))

            t.update()

    # compute mean of all metrics in summary
    outputs, labels = zip(*outputs_labels)
    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)
    summary = {metric: metrics[metric](outputs, labels)
               for metric in metrics}
    summary['loss'] = np.mean(losses)

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in summary.items())
    logging.info("- Evaluate metrics: " + metrics_string)

    return summary

def train_and_evaluate(model, train_dataloader, 
                       val_dataloader, optimizer, 
                       loss_fn, metrics, params, 
                       model_dir, restore_file=None, 
                       scheduler=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        load_checkpoint(restore_path, model, optimizer)

    best_val_primary_metric = 0.0
    for epoch in range(params["num_epochs"]):
        if scheduler:
            scheduler.step()

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params["num_epochs"]))

        # compute number of batches in one epoch (one full pass over the training set)
        logging.info("Train")
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        logging.info("Evaluate")
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)

        val_primary_metric = val_metrics[params["primary_metric"]]
        is_best = val_primary_metric >= best_val_primary_metric

        # Save weights
        # can't save sparse tensor
        state_dict = model.state_dict()
        #del state_dict["A_sparse"]
        #save_checkpoint({'epoch': epoch + 1,
        #                 'state_dict': state_dict,
        #                 'optim_dict' : optimizer.state_dict()},
        #                is_best=is_best,
        #                checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best " + params["primary_metric"])

            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            second_best_json_path = os.path.join(model_dir,     
                                                 "metrics_val_second_best_weights.json")

            if os.path.isfile(best_json_path):
                second_best_val_primary_metric = best_val_primary_metric
                # Copy former best val metrics into second best val metrics
                copyfile(best_json_path, second_best_json_path)

            best_val_primary_metric = val_primary_metric
            # Save best val metrics in a json file in the model directory
            save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        save_dict_to_json(val_metrics, last_json_path)