"""Run experiment"""
import logging
import os
import json
import datetime
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

from dpp.data.associations import load_diseases
from dpp.data.network import PPINetwork
from dpp.data.protein import load_drug_targets, load_essential_proteins
from dpp.experiments.experiment import Experiment
from dpp.util import Params, set_logger, prepare_sns

def jaccard(set_a, set_b):
    "Jaccard similarity between two sets"
    return len(set_a & set_b) / len(set_a | set_b)


class NetworkSimilarity(Experiment):
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
        
        logging.info("Loading networks...")
        self.networks = {name: PPINetwork(path) 
                         for name, path in self.params["ppi_networks"].items()}
    
  
        
    def compute_similarities(self):
        id_to_stats = {}
        
        for (name_a, name_b) in combinations(self.networks.keys(), 2):
            edges_a = set(self.networks[name_a].get_interactions())
            edges_b = set(self.networks[name_b].get_interactions())
            score =  len(edges_b - edges_a) / len(edges_b)
            print(f"{name_b}-{name_a}: {score}")

    
    def _run(self):
        """
        Run the experiment.
        """     
        id_to_stats = self.compute_similarities()
       