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
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

from pet_ct.util.util import hard_to_soft, flex_concat, place_on_cpu, get_batch_size
from pet_ct.data.report_transforms import split_impression_sections, word_tokenize


class Metrics:
    """
    """
    def __init__(self, metric_configs=[]):
        """
        args:
            metrics_fns (list, strings)   list of function names to use for eval
            break_ties (string) method to break ties (in probabilistic labels).
        """
        self.metric_configs = metric_configs

        # Dictionary mapping functions to
        self.metrics = defaultdict(dict)

        self.global_metrics = defaultdict(int)

        self.preds = defaultdict(list)
        self.targets = defaultdict(list)

        self.info = []

        self.precomputed_keys = None
        self.total_size = 0

    def add(self, preds, targets,
            info, precomputed_metrics={}):
        """
        Add a batches preds and targets. If you've already computed metrics for the batch
        (e.g. loss) you can pass in that value via precomputed metrics and it will update
        the average value in metrics. Note: must always provide the same precomputed
        metrics.
        args:
            preds    (list(tensor))    [(batch_size, ..., k)]
            targets     (list(tensor))    [(batch_size, ...,  1)]
            precomputed_metrics (dict) A dictionary of metric  values that have already
                been computed for the batch
        """
        # convert to list if only one element is passed in
        if type(preds) != dict:
            preds = {"primary": preds}
        if type(targets) != dict:
            targets = {"primary": targets}

        if preds.keys() != targets.keys():
            raise ValueError("Predictions and targets over different tasks.")

        tasks = list(preds.keys())
        batch_size = get_batch_size(list(targets.values())[0])
        for task in tasks:
            task_targets = targets[task]
            task_preds = preds[task]
            if(get_batch_size(task_targets) != get_batch_size(task_preds)):
                raise ValueError("preds must match targets in first dim.")

            self.preds[task].append(place_on_cpu(task_preds))
            self.targets[task].append(place_on_cpu(task_targets))

        self.info.extend(info)

        # include precomputed keys in global metrics
        if self.precomputed_keys is None:
            self.precomputed_keys = set(precomputed_metrics.keys())
        elif self.precomputed_keys != set(precomputed_metrics.keys()):
            raise ValueError("must always supply same precomputed metrics.")

        for key, value in precomputed_metrics.items():
            self.global_metrics[key] = ((self.total_size * self.global_metrics[key] +
                                         batch_size * value) /
                                        (batch_size + self.total_size))
        self.total_size += batch_size

    def compute(self):
        """
        Computes metrics on all
        """
        # call all metric_fns, detach since output has require grad
        for metric_config in self.metric_configs:
            self._compute_metric(**metric_config)

    def _compute_metric(self, fn, args={}, name=None, tasks=None,
                        is_primary=False, primary_task="primary"):
        """
        """
        name = name if name is not None else fn

        values = []
        for task in self.preds.keys():
            if tasks is not None and task not in tasks:
                continue

            total_value = 0
            total_size = 0

            all_preds = []
            all_targets = []
            for batch_preds, batch_targets in zip(self.preds[task], self.targets[task]):
                if type(batch_preds) is torch.Tensor:
                    # flatten dimensions
                    batch_preds = batch_preds.view(-1, batch_preds.shape[-1]).squeeze(-1)
                    batch_targets = batch_targets.view(-1).squeeze(-1)
                all_preds.append(batch_preds)
                all_targets.append(batch_targets)

            all_preds = flex_concat(all_preds, dim=0)
            all_targets = flex_concat(all_targets, dim=0)

            value = globals()[fn](all_preds, all_targets, **args)

            self.metrics[task][name] = value
            values.append(value)

            if is_primary and primary_task == task:
                self.primary_metric = self.metrics[task][name]

        self.global_metrics[name] = np.mean(values)

    def get_metric(self, metric, task=None):
        """
        """
        if task is None:
            return self.global_metrics[metric]
        else:
            return self.metrics[task][metric]


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
