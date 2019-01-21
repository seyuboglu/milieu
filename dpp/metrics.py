"""
General analysis code for our production-level trained code
"""
from __future__ import division
import pickle
import numpy as np
import csv
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import recall_score, precision_recall_curve, average_precision_score, roc_auc_score, roc_curve, precision_recall_curve


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def inverse_sample_mask(idx, l):
    mask = np.ones(l)
    mask[idx] = 0
    return np.array(mask, dtype=np.bool)    


def recall(y_true, y_pred, excluded_indices=[]): 
    no_train_mask = inverse_sample_mask(excluded_indices, y_true.shape[0])
    y_true_no_train = y_true[no_train_mask]
    y_pred_no_train = y_pred[no_train_mask]
    return recall_score(y_true_no_train, y_pred_no_train)


def recall_at(y_true, output_probs, k, excluded_indices=[]): 
    no_train_mask = inverse_sample_mask(excluded_indices, y_true.shape[0])
    y_true_no_train = y_true[no_train_mask]
    output_probs_no_train = output_probs[no_train_mask]
    sorted_nodes = np.argsort(output_probs_no_train)[::-1]
    y_pred_no_train = np.zeros((y_true_no_train.shape[0]))
    y_pred_no_train[sorted_nodes[:k]] = 1.0
    return recall_score(y_true_no_train, y_pred_no_train)


def auroc(y_true, output_probs, excluded_indices=[]): 
    no_train_mask = inverse_sample_mask(excluded_indices, y_true.shape[0])
    y_true_no_train = y_true[no_train_mask]
    output_probs_no_train = output_probs[no_train_mask]
    return roc_auc_score(y_true_no_train, output_probs_no_train)


def average_precision(y_true, output_probs, excluded_indices=[]):
    no_train_mask = inverse_sample_mask(excluded_indices, y_true.shape[0])
    y_true_no_train = y_true[no_train_mask]
    output_probs_no_train = output_probs[no_train_mask]
    return average_precision_score(y_true_no_train, output_probs_no_train)


def compute_ranking(scores):
    return rankdata(-scores, method='ordinal')


def positive_rankings(y_true, output_probs, excluded_indices): 
    no_train_mask = inverse_sample_mask(excluded_indices, y_true.shape[0])
    y_true_no_train = y_true[no_train_mask]
    output_probs_no_train = output_probs[no_train_mask]
    ranked_nodes = compute_ranking(output_probs_no_train)
    y_true_no_train = y_true_no_train.reshape((y_true_no_train.shape[0],))
    true_rankings = ranked_nodes[y_true_no_train.astype(bool)]
    return true_rankings
    