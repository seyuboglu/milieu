"""
"""
import os
import logging
import pickle
import json

from collections import defaultdict
import numpy as np 
import networkx as nx
from scipy.sparse import csr_matrix

from milieu.run import Process
from milieu.data.network import PPINetwork


class BuildNetworkMatrix(Process):
    """
    """

    def __init__(self, dir, params):
        """
        """
        super().__init__(dir, params)
        logging.info("Network Matrix Builder")
        logging.info("Sabri Eyuboglu  -- SNAP Group")
        logging.info("======================================")
        logging.info("Loading PPI Network...")
        self.network = PPINetwork(params["ppi_network"])
        self.deg_fn = params["deg_fn"]
        self.col_norm = params["col_norm"]
        self.row_norm = params["row_norm"]
        self.self_loops = params["self_loops"]
    
    def _run(self):
        """
        """
        self.ci_matrix = self.build_matrix()

    def save(self):
        """
        """
        logging.info("Saving matrix...")
        np.save(os.path.join(self.dir, "matrix.npy"), self.ci_matrix)
        # save protein to node to confirm indexes
        #with open('protein_to_node.pkl', 'wb') as f:
        #    pickle.dump(self.network.protein_to_node, f)
        np.save(os.path.join(self.dir, "protein_to_node.npy"), 
                self.network.protein_to_node)

    def build_matrix(self):
        """Builds a direct neighbor score matrix with optional normalization.
        """
        logging.info("Building matrix...")
        if self.self_loops:
            self.network.adj_matrix += np.identity(len(self.network))

        # Build vector of node degrees
        deg_vector = np.sum(self.network.adj_matrix, axis=1, keepdims=True)

        # Apply the degree function
        if self.deg_fn == 'log':
            # Take the natural log of the degrees. Add one to avoid division by zero
            deg_vector = np.log(deg_vector) + 1
        elif self.deg_fn == 'sqrt':
            # Take the square root of the degrees
            deg_vector = np.sqrt(deg_vector) 

        # Take the inverse of the degree vector
        inv_deg_vector = np.power(deg_vector, -1)

        # Build the complementarity matrix with sparse 
        ci_matrix = (csr_matrix((inv_deg_vector * self.network.adj_matrix).T) * 
                     csr_matrix(self.network.adj_matrix)).toarray()

        if(self.row_norm):
            # Normalize by the degree of the query node. (row normalize)
            ci_matrix = inv_deg_vector * ci_matrix
        
        if(self.col_norm):
            # Normalize by the degree of the disease node. (column normalize)
            ci_matrix = (ci_matrix.T * inv_deg_vector).T
        
        return ci_matrix 

def load_network_matrices(name_to_matrix, network=None):
    """
    Loads a set of ppi_matrices stored in numpy files (.npy).
    Also zeroes out the diagonal. Ensure that the degree of the 
    args:
        name_to_matrix    (dict) name to matrix file path
        netowrk 
    """
    ppi_matrices = {}
    for name, matrix_dir in name_to_matrix.items():
        matrix = np.load(os.path.join(matrix_dir, "matrix.npy"))
        protein_to_node = np.load(os.path.join(matrix_dir, "protein_to_node.npy"))
        assert(network is None or network.protein_to_node == protein_to_node)
        np.fill_diagonal(matrix, 0) 
        ppi_matrices[name] = matrix
    return ppi_matrices


def main(process_dir, overwrite, notify):
    with open(os.path.join(process_dir, "params.json")) as f:
        params = json.load(f)
    assert(params["process"] == "network_matrices")
    global exp
    exp = BuildNetworkMatrix(process_dir, params["process_params"])
    exp.run()
    exp.save()
