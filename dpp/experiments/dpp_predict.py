"""Run experiment"""
import logging
import os
import json
import datetime
import time
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import truncnorm, rankdata
from sklearn.model_selection import LeaveOneOut
import pandas as pd
from tqdm import tqdm

from dpp.data.associations import load_diseases
from dpp.data.network import PPINetwork
from dpp.experiments.experiment import Experiment
from dpp.methods.lci.lci_method import LCI
from dpp.util import Params, set_logger, string_to_list, fraction_nonzero


class DPPPredict(Experiment):
    """
    Class for running experiment that conducts enrichment of gene ontology terms in 
    pathways in the PPI network. 
    """
    def __init__(self, dir, params):
        """
        Constructor 
        Args: 
            dir (string) directory of the experiment to be run
        """
        super().__init__(dir, params)

        # Set the logger
        set_logger(os.path.join(self.dir, 'experiment.log'), level=logging.INFO, console=True)

        # Log title 
        logging.info("Disease Protein Prediction")
        logging.info("Sabri Eyuboglu  -- SNAP Group")
        logging.info("======================================")
        
        logging.info("Loading Disease Associations...")
        self.diseases_dict = load_diseases(self.params["diseases_path"], 
                                           self.params["disease_subset"],
                                           exclude_splits=['none'])
        
        logging.info("Loading Network...")
        self.network = PPINetwork(self.params["ppi_network"]) 

        self.method = globals()[self.params["method_class"]](self.network, 
                                                             self.diseases_dict, 
                                                             self.params["method_params"])
    
    def process_disease(self, disease):
        """
        """
        # compute method scores for disease
        disease_nodes = disease.to_node_array(self.network)
        scores = self.method.compute_scores(disease_nodes, None, disease)

        # zero out scores for disease_nodes
        scores[disease_nodes] = 0

        results = {self.network.get_proteins([node])[0]: score 
                   for node, score in enumerate(scores)}
        
        return disease, results

    def _run(self):
        """
        Run the experiment.
        """        
        results = []
        indices = []

        diseases = list(self.diseases_dict.values())
        diseases.sort(key=lambda x: x.split)
        if self.params["n_processes"] > 1:
            with tqdm(total=len(diseases)) as t: 
                p = Pool(self.params["n_processes"])
                for disease, results in p.imap(process_disease_wrapper, diseases):
                    results.append(results)
                    indices.append(disease.id)
                    t.update()
        else:
            with tqdm(total=len(diseases)) as t: 
                for disease in diseases:
                    disease, result = self.process_disease(disease)
                    results.append(result)
                    indices.append(disease.id)
                    t.update()
        
        self.results = pd.DataFrame(results, index=indices)


    def save_results(self, summary=True):
        """
        Saves the results to a csv using a pandas Data Fram
        """
        print("Saving Results...")
        self.results.to_csv(os.path.join(self.dir, 'predictions.csv'))
    
    def load_results(self):
        """
        Loads the results from a csv to a pandas Data Frame.
        """
        print("Loading Results...")
        self.results = pd.read_csv(os.path.join(self.dir, 'predictions.csv'))
    

def process_disease_wrapper(disease):
    return exp.process_disease(disease)


def main(process_dir, overwrite, notify):
    with open(os.path.join(process_dir, "params.json")) as f:
        params = json.load(f)
    assert(params["process"] == "dpp_predict")
    exp = DPPPredict(process_dir, params["process_params"])
    if exp.is_completed():
        exp.load_results()
    elif exp.run():
        exp.save_results()
    exp.plot_results()
