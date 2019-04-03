"""
Defines metric functions of form
    def metric_name(output_batch, labels_batch):
Where output_batch and labels_batch are torch tensors on the cpu.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
import sklearn.metrics as skl
import torch

from  dpp.metrics import recall_score


class Metrics:
    """
    """
    def __init__(self, metric_configs):
        """
        args:
            metrics_fns (list, strings)   list of function names to use for eval
        """
        self.metric_configs = metric_configs
        # Dictionary mapping functions to
        self.metrics = defaultdict(int)

        self.outputs = torch.Tensor()
        self.probs = torch.Tensor()
        self.labels = torch.Tensor()

        self.precomputed_keys = None
 
    def add(self, batch_outputs, batch_labels, precomputed_metrics={}):
        """
        Add a batches outputs and labels. If you've already computed metrics for the batch
        (e.g. loss) you can pass in that value via precomputed metrics and it will update
        the average value in metrics. Note: must always provide the same precomputed 
        metrics. 
        args:
            batch_outputs    (tensor)    (batch_size, k)
            batch_labels     (tensor)    (batch_size, 1)
            precomputed_metrics (dict) A dictionary of metric  values that have already
                been computed for the batch
        """
        if(batch_labels.shape[0] != batch_outputs.shape[0]):
            raise ValueError("batch_probs must match batch_labels in first dim.")
            
        if self.precomputed_keys is None:
            self.precomputed_keys = set(precomputed_metrics.keys()) 
        elif self.precomputed_keys != set(precomputed_metrics.keys()):
            raise ValueError("must always supply same precomputed metrics.")

        batch_size = batch_labels.shape[0]
        total_size = self.labels.shape[0]
        
        batch_probs = torch.nn.functional.softmax(batch_outputs, dim=1)
        self.outputs = torch.cat([self.outputs, batch_outputs.cpu().detach()])
        self.probs = torch.cat([self.probs, batch_probs.cpu().detach()])
        self.labels = torch.cat([self.labels, batch_labels.float().cpu().detach()])
        
        for key, value in precomputed_metrics.items():
            self.metrics[key] = ((total_size * self.metrics[key] + batch_size * value) / 
                                 (batch_size + total_size))

    def compute(self):
        """
        Computes metrics on all 
        """
        # call all metric_fns, detach since output has require grad
        for metric_config in self.metric_configs:
            metric_fn = metric_config["fn"]
            self.metrics[metric_fn] = globals()[metric_fn](self.probs,
                                                                     self.labels)


def accuracy(probs, targets):
    """
    Computes accuracy between output and labels for k classes. Targets with class -1 are
    ignored.
    args:
        probs    (tensor)    (size, k)  2-d array of class probabilities
        labels     (tensor)    (size, 1) 1-d array of correct class indices
    """
    probs = torch.nn.functional.softmax(probs, dim=1)
    probs = probs.numpy()
    targets = targets.numpy()

    pred = np.argmax(probs, axis=1)

    # ignore -1
    pred = pred[(targets != -1).squeeze()]
    targets = targets[targets != -1]

    return np.sum(pred == targets) / targets.size


def roc_auc(probs, labels):
    """
    Computes the area under the receiving operator characteristic between output probs
    and labels for k classes.
    Source: https://github.com/HazyResearch/metal/blob/master/metal/utils.py
    args:
        probs    (tensor)    (size, k)
        labels     (tensor)    (size, 1)
    """
    probs = torch.nn.functional.softmax(probs, dim=1)
    probs = probs.numpy()
    # Convert labels to one-hot indicator format, using the k inferred from probs
    labels = hard_to_soft(labels, k=probs.shape[1]).numpy()
    return skl.roc_auc_score(labels, probs)


def precision(probs, labels):
    """
    Computes the precision score between output and labels for k classes.
    args:
        probs    (tensor)    (size, k)
        labels     (tensor)    (size, 1)
    """
    probs = torch.nn.functional.softmax(probs, dim=1)
    probs = probs.numpy()
    labels = labels.numpy()

    pred = np.argmax(probs, axis=1)
    return skl.precision_score(labels, pred)


def recall(probs, labels):
    """
    Computes the recall score between output and labels for k classes.
    args:
        probs    (tensor)    (size, k)
        labels     (tensor)    (size, 1)
    """
    probs = torch.nn.functional.softmax(probs, dim=1)
    probs = probs.numpy()
    labels = labels.numpy()

    pred = np.argmax(probs, axis=1)

    return skl.recall_score(labels, pred)


def f1_score(probs, labels):
    """
    Computes the f1 score between output and labels for k classes.
    args:
        probs    (tensor)    (size, k)
        labels     (tensor)    (size, 1)
    """
    probs = torch.nn.functional.softmax(probs, dim=1)
    probs = probs.numpy()
    labels = labels.numpy()

    pred = np.argmax(probs, axis=1)
    return skl.f1_score(labels, pred, pos_label=1)


def recall_at_100(probs, labels):
    """
    """
    k = 100
    probs = probs.numpy()
    labels = labels.numpy()

    N, M = probs.shape
    argsort_output = np.argsort(probs, axis=1)
    rows = np.column_stack((np.arange(N),) * k)
    cols = argsort_output[:, (M - k):]
    binary_output = np.zeros_like(probs)
    binary_output[rows, cols] = 1

    recall = np.mean([recall_score(labels[row, :], 
                      binary_output[row, :]) for row in range(N)])

    return recall


def hard_to_soft(Y_h, k):
    """Converts a 1D tensor of hard labels into a 2D tensor of soft labels
    Source: MeTaL from HazyResearch, https://github.com/HazyResearch/metal/blob/master/metal/utils.py
    Args:
        Y_h: an [n], or [n,1] tensor of hard (int) labels in {1,...,k}
        k: the largest possible label in Y_h
    Returns:
        Y_s: a torch.FloatTensor of shape [n, k] where Y_s[i, j-1] is the soft
            label for item i and label j
    """
    Y_h = Y_h.clone()
    Y_h = Y_h.squeeze()
    assert Y_h.dim() == 1
    assert (Y_h >= 0).all()
    assert (Y_h < k).all()
    n = Y_h.shape[0]
    Y_s = torch.zeros((n, k), dtype=Y_h.dtype, device=Y_h.device)
    for i, j in enumerate(Y_h):
        Y_s[i, int(j)] = 1.0
    return Y_s

