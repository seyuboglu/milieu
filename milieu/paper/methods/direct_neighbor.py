"""
Provides method for running random walks on disease pathways.
"""

import numpy as np 

from milieu.paper.methods.method import DPPMethod


class DirectNeighbor(DPPMethod):
    """

    """
    def __init__(self, ppi_network, diseases, params):
        """

        """
        super().__init__(ppi_network, diseases, params)

    def compute_scores(self, train_nodes, disease):
        adjacency_matrix = self.network.adj_matrix

        assoc_neighbors = np.sum(adjacency_matrix[:, train_nodes], axis=1)
        total_neighbors = np.sum(adjacency_matrix, axis=1)
        scores = assoc_neighbors / (total_neighbors + 1e-50)
        return scores  


