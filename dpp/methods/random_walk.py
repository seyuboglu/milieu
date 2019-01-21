"""
Provides method for running random walks on disease pathways.
"""

import numpy as np 
import networkx as nx

from dpp.methods.method import DPPMethod


class RandomWalk(DPPMethod):
    """

    """
    def __init__(self, ppi_network, diseases, params):
        """

        """
        super().__init__(ppi_network, diseases, params)

        self.alpha = params["alpha"]

    def compute_scores(self, train_nodes, disease):
        """
        Computes the scores 
        """
        train_nodes = set(train_nodes)
        training_personalization = {node: (1.0 / len(train_nodes) 
                                    if node in train_nodes else 0) 
                                    for node in self.network.nx.nodes()}
        page_rank = nx.pagerank(self.network.nx, 
                                alpha=self.alpha, 
                                personalization=training_personalization)
        scores = np.array([page_rank.get(node, 0) for node in self.network.nx.nodes()])   
        return scores 
