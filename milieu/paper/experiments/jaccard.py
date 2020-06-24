"""Run experiment"""
import logging
import os
import json
import datetime
from collections import defaultdict, Counter
from multiprocessing import Pool

import numpy as np
from scipy.stats import spearmanr, pearsonr, ttest_ind, ttest_rel
from scipy.sparse import csr_matrix
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

from milieu.data.associations import load_diseases, build_disease_matrix
from milieu.data.network_matrices import load_network_matrices
from milieu.data.network import Network
from milieu.data.protein import load_drug_targets, load_essential_proteins
from milieu.paper.experiments.experiment import Experiment
from milieu.util.util import set_logger, prepare_sns


class JaccardComparison(Experiment):
    """
    Class for running experiment that compute simple network metrics for
    disease pathways. 
    """

    def __init__(self, dir, params):
        """
        Constructor 
        Args: 
            dir (string) directory of the experiment to be run
        """
        super().__init__(dir, params)

        # set the logger
        set_logger(
            os.path.join(self.dir, "experiment.log"), level=logging.INFO, console=True
        )

    def _run(self):
        """
        Run the experiment.
        """
        logging.info("Loading network...")
        network = Network(self.params["ppi_network"])

        logging.info("Loading molecule associations...")
        associations = {}
        for association_path in self.params["association_paths"]:
            dct = load_diseases(association_path)
            associations.update(dct)

        association_matrix, _ = build_disease_matrix(associations, network)

        association_jaccard = compute_jaccard(association_matrix.T)

        mi_matrix = mi_matrix = load_network_matrices(
            {"mi": self.params["mi_dir"]}, network=network
        )["mi"]

        mi_values = mi_matrix[np.triu_indices(mi_matrix.shape[0], k=1)]
        adj_values = network.adj_matrix[
            np.triu_indices(network.adj_matrix.shape[0], k=1)
        ]
        jaccard_values = association_jaccard[
            np.triu_indices(association_jaccard.shape[0], k=1)
        ]

        k = adj_values.sum().astype(int)
        statistic, pvalue = ttest_rel(
            jaccard_values[np.argpartition(mi_values, -k)[-k:]],
            jaccard_values[np.argpartition(adj_values, -k)[-k:]],
        )

        metrics = {
            "test": "ttest_rel",
            "statistic": statistic,
            "pvalue": pvalue,
            "mi_mean": jaccard_values[np.argpartition(mi_values, -k)[-k:]].mean(),
            "adj_mean": jaccard_values[np.argpartition(adj_values, -k)[-k:]].mean(),
        }

        with open(os.path.join(self.dir, "results.json"), "w") as f:
            json.dump(metrics, f, indent=4)


def compute_jaccard(matrix):
    """
    Computes the pairwise jaccard similarity between 
    :param matrix: (nd.array) an NxD matrix where N is the # of sets and D is
        the maximum cardinality of the sets.  
    """
    intersection = csr_matrix(matrix).dot(csr_matrix(matrix.T)).todense()
    union = np.zeros_like(intersection)
    union += matrix.sum(axis=1, keepdims=True)
    union += matrix.sum(axis=1, keepdims=True).T
    union -= intersection
    jaccard = np.array(np.nan_to_num(intersection / union, 0))
    return jaccard


def test_jaccard(matrix):
    # test jaccard
    jaccard = compute_jaccard(matrix)

    from sklearn.metrics import jaccard_score

    for _ in range(1000):
        i = np.random.randint(0, jaccard.shape[0])
        j = np.random.randint(0, jaccard.shape[1])
        if i == j:
            continue

        computed = jaccard[i, j]
        value = jaccard_score(matrix[i, :], matrix[j, :])

        if computed != value:
            raise ValueError("Failed")

    print("passed")


def main(process_dir, overwrite, notify):
    with open(os.path.join(process_dir, "params.json")) as f:
        params = json.load(f)
    assert params["process"] == "jaccard_comparison"
    global exp
    exp = JaccardComparison(process_dir, params["process_params"])
    if exp.is_completed():
        exp.load_results()
    elif exp.run():
        exp.save_results()
