import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from nimfa import Pmf as PMF

import seaborn as sns
import matplotlib.pyplot as plt

from dpp.methods.method import DPPMethod


class MatrixFactorization(DPPMethod):
    
    def __init__(self, network, diseases, params):
        """
        """
        super().__init__(network, diseases, params)

        self.dti_matrix = np.zeros((len(diseases), len(network))) # num_drugs x num_proteins
        self.drug_to_idx = {}
        for idx, drug in enumerate(diseases.values()):
            target_nodes = drug.to_node_array(network)
            self.dti_matrix[idx, target_nodes] = 1
            self.drug_to_idx[drug.id] = idx
    
    def compute_scores(self, train_pos, disease):
        """
        """
        # copy matrix so we don't affect the main matrix
        dti_matrix = np.copy(self.dti_matrix)
        
        # mask out the test proteins 
        drug_idx = self.drug_to_idx[disease.id]
        dti_matrix[drug_idx, :] = 0
        dti_matrix[drug_idx, train_pos] = 1

        # fit the PMF
        pmf = PMF(dti_matrix, seed="random_vcol", rank=self.params["rank"], 
                  max_iter=self.params["max_iter"], rel_error=self.params["rel_error"])
        pmf_fit = pmf()
        
        # get scores 
        scores = np.array(pmf_fit.fitted()[drug_idx, :]).squeeze()

        return scores 
    

        
