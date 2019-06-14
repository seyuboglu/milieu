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
from dpp.data.protein import load_drug_targets, load_essential_proteins
from dpp.experiments.experiment import Experiment
from dpp.util import Params, set_logger, prepare_sns


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
        
    
    def plot_drug_weight_dist(self, protein_sets, save="weight_dist.pdf"):
        """
        """
        
        prepare_sns(sns, kwargs={"font_scale": 1.4,
                         "rc": {'figure.figsize':(6, 4)}})
        for name, proteins in protein_sets.items():
            sns.distplot(self.ci_weights_norm[proteins], 
                 kde=False, hist=True, norm_hist=True, bins=50, 
                 hist_kws={"range":(-0.25, 0.75),
                           "alpha": 0.8},
                         label=name)

        sns.despine()
        plt.xscale('linear')
        plt.yscale('linear')
        plt.legend()
        plt.xlabel(r"Degree-normalized LCI weight, $\frac{w_z}{\sqrt{d_z}}$")
        plt.ylabel("Density")
        plt.tight_layout()
        
        if save is not None:
            plt.savefig(os.path.join(self.dir, "_figures", save))

        
class FunctionalEnrichmentAnalysis(Experiment):
    """
    """
    
    def __init__(self, dir, params):
        """
        """
        super().__init__(dir, params)
        
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
        
        logging.info("Loading enrichment study...")
        geneid2go = read_ncbi_gene2go("data/go/gene2go.txt", taxids=[9606])
        obodag = GODag("data/go/go-basic.obo")
        self.go_study = GOEnrichmentStudy(self.network.get_proteins(),
                                          geneid2go,
                                          obodag, 
                                          propagate_counts = True,
                                          alpha = 0.05,
                                          methods = ['fdr_bh'])

    
    def run_study(self):
        """
        """
        top_nodes = np.argsort(self.ci_weights_norm)[-self.params["top_k"]:]
        top_proteins = self.network.get_proteins(top_nodes)
        self.raw_results = self.go_study.run_study(set(top_proteins))  
    
    def to_csv(self):
        """
        """
        self.results = []
        for r in self.raw_results:
            self.results.append({
                "name": r.name,
                "pvalue": r.p_fdr_bh,
                "goterm_id": r.goterm.id
            })
        self.results = sorted(self.results, key = lambda x: x["pvalue"])
        
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(os.path.join(self.dir, "all_terms.csv"))            
            
        
        
        
        
        

class EssentialGeneAnalysis(Experiment):
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

                
        logging.info("Loading essential genes...")
        self.essential_proteins = load_essential_proteins(params["essential_genes_path"])
        self.essential_nodes = self.network.get_nodes(self.essential_proteins)
        self.non_essential_nodes = [node for node in self.network.get_nodes()
                                    if node not in self.essential_nodes]
        
        self.essential_array = np.zeros(len(self.network))
        self.essential_array[self.essential_nodes] = 1

        
    def compute_weight_stats(self, nodes=None, norm=True):
        """
        """
        weights = self.ci_weights_norm if norm else self.ci_weights
        if nodes is None:
            nodes = np.arange(len(self.network))
        return {
            "mean": np.mean(weights[nodes]),
            "median": np.median(weights[nodes]),
            "std": np.std(weights[nodes])
        }

    def compute_frac_essential(self, nodes): 
        """
        """
        return np.mean(self.essential_array[nodes])
         
        
    def plot_weight_dist(self, node_sets):
        """
        """
        for name, nodes in node_sets.items():
            sns.distplot(self.ci_weights[nodes], 
                 kde=False, hist=True, norm_hist=True, bins=15, 
                 hist_kws={"range":(-0.4, 0.8)}, label=name)

        plt.xscale('linear')
        plt.yscale('linear')
        plt.legend()
        plt.xlabel(r"$\frac{w_k}{\sqrt{d_k}}$")
        plt.ylabel("# of proteins [normalized]")
        