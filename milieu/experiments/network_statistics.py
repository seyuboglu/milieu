"""Run experiment"""
import logging
import os
import json
import datetime
from collections import defaultdict, Counter
from multiprocessing import Pool

import numpy as np
from scipy.stats import spearmanr
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

from milieu.data.associations import load_diseases
from milieu.data.network import PPINetwork
from milieu.data.protein import load_drug_targets, load_essential_proteins
from milieu.experiments.experiment import Experiment
from milieu.util import Params, set_logger, prepare_sns


class NetworkStatistics(Experiment):
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
        set_logger(os.path.join(self.dir, 'experiment.log'), 
                   level=logging.INFO, console=True)

        logging.info("Loading disease associations...")
        self.diseases_dict = load_diseases(self.params["diseases_path"], 
                                           self.params["disease_subset"],
                                           exclude_splits=['none'])
        
        logging.info("Loading network...")
        self.network = PPINetwork(self.params["ppi_network"]) 
        self.degrees = np.array(list(dict(self.network.nx.degree()).values()))
    
    def compute_cut_ratio(self, nodes):
        """
        Computes cut ratio as defined in https://cs.stanford.edu/~jure/pubs/comscore-icdm12.pdf. 
        """
        cut_size = nx.algorithms.cuts.cut_size(self.network.nx, nodes)
        cut_ratio = cut_size / (len(nodes) * (len(self.network) - len(nodes)))
        return cut_ratio
        
    def compute_stats(self):
        id_to_stats = {}
        
        for disease_id, disease in tqdm(self.diseases_dict.items()):
            
            stats = {}
            nodes = disease.to_node_array(self.network)
            subgraph = self.network.nx.subgraph(nodes)
            
            stats["num_nodes"] = len(nodes)
            stats["num_proteins"] = len(disease)
            
            stats["density"] = nx.density(subgraph)
            stats["conductance"] = nx.algorithms.cuts.conductance(self.network.nx, nodes)
            stats["cut_ratio"] = self.compute_cut_ratio(nodes)
            
            id_to_stats[disease_id] = stats
        
        return id_to_stats
        
  
    
    def _run(self):
        """
        Run the experiment.
        """     
        id_to_stats = compute_stats()
        df = pd.DataFrame.from_dict(id_to_stats, orient="index")
        df.to_csv(os.path.join(self.dir, "results.csv"))