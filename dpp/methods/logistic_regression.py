"""
Provides method for running logistic regression for disease
pathway prediction. 
"""
from random import shuffle

import numpy as np
from sklearn.linear_model import LogisticRegression as SKLogisticRegression

from dpp.methods.method import DPPMethod




class LogisticRegression(DPPMethod):
    """

    """
    def __init__(self, ppi_network, diseases, params):
        """

        """
        super().__init__(ppi_network, diseases, params)

        self.l2_reg = params["l2_reg"]
        self.build_embedding_feature_matrix(params["features_path"])
    
    def build_embedding_feature_matrix(self, embedding_path): 
        """ Builds a numpy matrix for a node embedding encoded in the embeddingf ile
        passed in. Row indices are given by protein_to_node dictionary passed in. 
        Args:
            protein_to_node (dictionary)
            embedding_path (string)
        """
        with open(embedding_path) as embedding_file:
            # get  
            line = embedding_file.readline()
            n_nodes, n_dim = map(int, line.split(" "))
            feature_matrix = np.empty((len(self.network), n_dim))

            for index in range(n_nodes):
                line = embedding_file.readline()
                line_elements = line.split(" ")
                protein = int(line_elements[0])

                if protein not in self.network.protein_to_node: 
                    continue  
                    
                node_embedding = list(map(float, line_elements[1:]))
                feature_matrix[self.network.get_node(protein), :] = node_embedding
        self.feature_matrix = feature_matrix

    def compute_scores(self, train_nodes, disease):
        """
        Computes the scores 
        """
        X = self.feature_matrix
        N, D = X.shape

        # Y: Build 
        Y = np.zeros(N)
        Y[train_nodes] = 1

        # Get sample of negative examples
        train_neg = self.get_negatives(Y, len(train_nodes))
        train_nodes = np.concatenate((train_nodes, train_neg))
        Y_train = Y[train_nodes]
        X_train = X[train_nodes, :]

        # Model 
        model = SKLogisticRegression(C=1.0 / self.l2_reg, solver="liblinear")

        # Run training 
        model.fit(X_train, Y_train)
        scores = model.predict_proba(X)[:, 1]   
        return scores 

    def get_negatives(self, Y, n_neg):
        """ Generate n_neg indices for negative examples
        excluding examples already positive in Y. 
        """
        n = Y.shape[0]
        n_pos = np.sum(np.sum(Y))
        neg_indices = np.random.choice(range(n), 
                                       size=int(n_neg), 
                                       replace=False, 
                                       p=(1 - Y) / (n - n_pos))                             
        return neg_indices 
