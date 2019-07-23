""" 
Metrics for classification.
"""

import torch 
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score 


def recall_at_100(output_batch, labels_batch):
    """
    """
    k = 100
    output_batch = output_batch.numpy()
    labels_batch = labels_batch.numpy()

    N, M = output_batch.shape
    argsort_output = np.argsort(output_batch, axis=1)
    rows = np.column_stack((np.arange(N),) * k)
    cols = argsort_output[:, (M - k):]
    binary_output = np.zeros_like(output_batch)
    binary_output[rows, cols] = 1

    recall = np.mean([recall_score(labels_batch[row, :], 
                      binary_output[row, :]) for row in range(N)])

    return recall

metrics = {
    "recall_at_100": recall_at_100
}
