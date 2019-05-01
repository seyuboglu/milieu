"""Run experiment"""
import logging
import os
import json
import datetime
import time
import pickle
from collections import defaultdict, Counter
from multiprocessing import Pool

import numpy as np
from scipy.stats import spearmanr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from tqdm import tqdm
from goatools.obo_parser import GODag
from goatools.associations import read_ncbi_gene2go
from goatools.go_enrichment import GOEnrichmentStudy


from dpp.data.associations import load_diseases
from dpp.data.network import PPINetwork
from dpp.data.drug import load_drug_targets
from dpp.experiments.experiment import Experiment
from dpp.util import Params, set_logger


class DrugTarget(Experiment):
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
        
        logging.info("Loading weights...")
        with open(os.path.join(params["model_path"], "models", "models.tar"), "rb") as f:
            split_to_model = pickle.load(f)
            
        self.ci_weights = ci_weights = np.mean([model['ci_weight'][0, 0].numpy() 
                                                for model in split_to_model.values()], axis=0)
        self.ci_weights_norm = self.ci_weights / np.sqrt(self.degrees)

                
        logging.info("Loading drugs...")
        self.drug_to_targets = load_drug_targets(params["drug_targets_path"])
        
    
    def compute_drug_counts(self):
        """
        Get np.array num_drugs where num_drugs[u] gives the count of drugs that target node u. 
        """
        protein_to_drug_count = Counter()
        for drug, targets in self.drug_to_targets.items():
            for target in targets:
                protein_to_drug_count[target] += 1
        node_to_drug_count = {self.network.get_node(protein): count 
                              for protein, count 
                              in protein_to_drug_count.items() 
                              if protein in self.network}
        num_drugs = np.zeros(len(self.network))
        num_drugs[list(node_to_drug_count)] = list(node_to_drug_count.values())
        
        self.drug_counts = np.array(num_drugs)
    
    def compute_weight_stats(self, proteins=None):
        """
        """
        if proteins is None:
            proteins = np.arange(len(self.network))
        return {
            "mean": np.mean(self.ci_weights_norm[proteins]),
            "median": np.median(self.ci_weights_norm[proteins]),
            "std": np.std(self.ci_weights_norm[proteins])
        }
    
    def compute_frac_targets(self, proteins=None):
        """
        """
        if proteins is None:
            proteins = np.arange(len(self.network))
        
        return np.mean((self.drug_counts > 0)[proteins])

    def frac_targets_ks_test(self, proteins_a, proteins_b):
        targets_a = (self.drug_counts > 0)[proteins_a]
        targets_b = (self.drug_counts > 0)[proteins_b]
        return ks_2samp(targets_a, targets_b)
    
    
    def _run(self):
        """
        Run the experiment.
        """     
        results = {"norm_weight": {},
                   "frac_targets": {}}
        self.compute_drug_counts()
        
        target_proteins = np.where(self.drug_counts != 0)
        not_target_proteins = np.where(self.drug_counts == 0)
        
        results["norm_weight"]["all"] = self.compute_weight_stats()
        results["norm_weight"]["target"] = self.compute_weight_stats(target_proteins)
        results["norm_weight"]["not_target"] = self.compute_weight_stats(not_target_proteins)
        
        top_proteins = np.argsort(self.ci_weights_norm)[-self.params["top_k"]:]
        bottom_proteins = np.argsort(self.ci_weights_norm)[:-self.params["top_k"]]
        
        results["frac_targets"]["top"] = self.compute_frac_targets(top_proteins)
        results["frac_targets"]["bottom"] = self.compute_frac_targets(bottom_proteins)
        
        results["frac_targets"]["pvalue"] = self.frac_targets_ks_test(top_proteins, bottom_proteins).pvalue
        
        with open(os.path.join(self.dir, "results.json"), 'w') as f: 
            json.dump(results, f, indent=4)
        
    
    def plot_drug_weight_dist(self, protein_sets):
        """
        """
        for name, proteins in protein_sets.items():
            sns.distplot(self.ci_weights_norm[proteins], 
                 kde=False, hist=True, norm_hist=True, bins=15, 
                 hist_kws={"range":(-0.4, 0.8)}, label=name)

        plt.xscale('linear')
        plt.yscale('linear')
        plt.legend()
        plt.xlabel(r"$\frac{w_k}{\sqrt{d_k}}$")
        plt.ylabel("# of proteins [normalized]")
        

    def save_results(self, summary=True):
        """
        Saves the results to a csv using a pandas Data Fram
        """
        print("Saving Results...")
        self.results.to_csv(os.path.join(self.dir, 'results.csv'))

        #if self.params["save_enrichment_results"]:
        #    with open(os.path.join(self.dir,'outputs.pkl'), 'wb') as f:
        #        pickle.dump(self.outputs, f)
            
    def load_results(self):
        """
        Loads the results from a csv to a pandas Data Frame.
        """
        print("Loading Results...")
        self.results = pd.read_csv(os.path.join(self.dir, 'results.csv'))
    

def main(process_dir, overwrite, notify):
    with open(os.path.join(process_dir, "params.json")) as f:
        params = json.load(f)
    assert(params["process"] == "go_enrichment")
    global exp
    exp = GOEnrichment(process_dir, params["process_params"])
    if exp.is_completed():
        exp.load_results()
    elif exp.run():
        exp.save_results()
    exp.plot_results()



