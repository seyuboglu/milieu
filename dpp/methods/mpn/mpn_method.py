"""
Provides method for running random walks on disease pathways.
"""

import numpy as np 
import scipy.sparse as sp
import torch

from dpp.methods.method import DPPMethod


class MPN(DPPMethod):
    """

    """
    def __init__(self, network, diseases, params):
        """

        """
        super().__init__(network, diseases, params)

        adj_matrix = network.adj_matrix 

        deg_vec = adj_matrix.sum(axis=1, keepdims=True)
        if params["deg_fn"] == 'log':
            deg_vec = np.log(deg_vec) + 1
        elif params["deg_fn"] == 'sqrt':
            deg_vec = np.sqrt(deg_vec) 
        
        # take the inverse of the degree vector
        inv_deg_vec = np.power(deg_vec, -1)

        if params["norm_mutual"]:
            mpn_matrix = (sp.csr_matrix((inv_deg_vec * adj_matrix).T) * 
                          sp.csr_matrix(adj_matrix)).toarray()
        else:
            mpn_matrix = (sp.csr_matrix(adj_matrix) * sp.csr_matrix(adj_matrix)).toarray()

        if params["norm_in"]:
            mpn_matrix = (mpn_matrix.T * inv_deg_vec).T
        
        if params["norm_out"]:
            mpn_matrix = inv_deg_vec * mpn_matrix
        
        self.mpn_matrix = mpn_matrix

    def compute_scores(self, train_nodes, disease):
        """
        Computes the scores 
        """
        return np.sum(self.mpn_matrix[train_nodes, :])
