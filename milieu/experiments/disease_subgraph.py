"""Run experiment"""

import argparse
import logging
import os
import datetime
import json
from multiprocessing import Pool
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt 
from networkx.readwrite.adjlist import write_adjlist
from scipy.stats import truncnorm, rankdata
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from milieu.data.associations import load_diseases
from milieu.data.network import PPINetwork
from milieu.data.network_matrices import load_network_matrices
from milieu.experiments.experiment import Experiment
from milieu.util import (Params, set_logger, prepare_sns, string_to_list, 
                      compute_pvalue, build_degree_buckets, list_to_string, load_mapping)


class DiseaseSubgraph(Experiment):
    """
    Class for running experiment that assess the significance of a network metric
    between disease proteins. Uses the method described in Guney et al. for generating
    random subgraph. 
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
        logging.info("Metric Significance of Diseases in the PPI Network")
        logging.info("Sabri Eyuboglu  -- SNAP Group")
        logging.info("======================================")
        logging.info("Loading Disease Associations...")
        self.diseases = load_diseases(self.params["diseases_path"], 
                                      self.params["disease_subset"],
                                      exclude_splits=['none'])
        
        logging.info("Loading Network...")
        self.network = PPINetwork(self.params["ppi_network"]) 

        logging.info("Loading Predictions...")
        self.method_to_preds = {name: pd.read_csv(os.path.join(preds, "predictions.csv"), 
                                                  index_col=0) 
                                for name, preds in self.params["method_to_preds"].items()}
        
        logging.info("Loading Protein Data...")
        self.field_to_protein_data = {field: load_mapping(path=config["path"],
                                                          **config["args"]) 
                                      for field, config 
                                      in self.params["field_to_protein_data"].items()} 

    def compute_disease_subgraph(self, disease):
        """ Get the disease subgraph of 
        Args:
            disease: (Disease) A disease object
        """
        node_to_roles = {}
        disease_nodes = disease.to_node_array(self.network)
        for disease_node in disease_nodes:
            node_to_roles[disease_node] = "disease"
        
        disease_node_to_nbrs = {node: set(self.network.nx.neighbors(node)) 
                                for node in disease_nodes}

        for method, preds in self.method_to_preds.items():
            top_pred_proteins = set(map(int, preds.loc[disease.id]
                                                  .sort_values(ascending=False)
                                                  .index[:self.params["num_preds"]]))
            top_pred_nodes = self.network.get_nodes(top_pred_proteins)

            for pred_node in top_pred_nodes:
                if pred_node not in node_to_roles:
                    node_to_roles[pred_node] = f"pred_{method}"
                pred_nbrs = set(self.network.nx.neighbors(pred_node)) 
                for disease_node in disease_nodes:
                    disease_nbrs = disease_node_to_nbrs[disease_node]
                    common_nbrs = disease_nbrs & pred_nbrs
                    for common_nbr in common_nbrs:
                        if common_nbr not in node_to_roles:
                            node_to_roles[common_nbr] = f"common_pred_{method}" 
        
        # the set of nodes intermediate between nodes in the 
        for a, node_a in enumerate(disease_nodes):
            for b, node_b in enumerate(disease_nodes):
                # avoid repeat pairs
                if a >= b:
                    continue
                common_nbrs = disease_node_to_nbrs[node_a] & disease_node_to_nbrs[node_b]
                for common_nbr in common_nbrs:
                    if common_nbr not in node_to_roles:
                        node_to_roles[common_nbr] = "common_disease"
        
        # get induced subgraph 
        subgraph = self.network.nx.subgraph(node_to_roles.keys())

        return subgraph, node_to_roles

    def write_subgraph(self, disease, node_to_roles, subgraph, delimiter='\t'):
        """
        """
        directory = os.path.join(self.dir, 'diseases', disease.id)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(os.path.join(directory, f"subgraph_{disease.id}.txt"), "w") as f:
            f.write(delimiter.join(["node_1", "node_2", "roles"]) + '\n')
            for edge in subgraph.edges():
                items = [str(edge[0]), str(edge[1])]

                # dd interaction type
                roles = node_to_roles[edge[0]] + "-" + node_to_roles[edge[1]]
                items.append(roles)

                f.write(delimiter.join(items) + '\n')

    
    def write_protein_data(self, disease, node_to_roles):
        """
        """
        directory = os.path.join(self.dir, 'diseases', disease.id)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        protein_data = []
        for node, roles in node_to_roles.items():
            protein_id = self.network.get_proteins([node])[0]
            node_dict = {
                "node_id": node, 
                "protein_id": protein_id,
                "role": roles,
                "degree": self.network.nx.degree(node)
            }
            
            for field, data in self.field_to_protein_data.items():
                if not ("weight" in field and "common" not in roles):
                    node_dict[field] = data.get(protein_id, "")
            protein_data.append(node_dict)
        
        df = pd.DataFrame(protein_data)
        df = df.set_index('node_id')
        df.to_csv(os.path.join(directory, f"data_{disease.id}.csv"))
            

    def process_disease(self, disease):
        """
        Generates null model for disease and computes 
        Args:
            disease (Disease) the current disease 
        """
        subgraph, node_to_roles = self.compute_disease_subgraph(disease)

        disease_directory = os.path.join(self.dir, 'diseases', disease.id)
        if not os.path.exists(disease_directory):
            os.makedirs(disease_directory)

        self.write_subgraph(disease, node_to_roles, subgraph)
        self.write_protein_data(disease, node_to_roles)


    def _run(self):
        """
        Run the experiment.
        """

        logging.info("Running Experiment...")
        self.results = []

        if self.params["n_processes"] > 1:
            with tqdm(total=len(self.diseases)) as t: 
                p = Pool(self.params["n_processes"])
                for results in p.imap(process_disease_wrapper, self.diseases.values()):
                    self.results.append(results)
                    t.update()
        else:
            with tqdm(total=len(self.diseases)) as t: 
                for disease in self.diseases.values():
                    results = self.process_disease(disease)
                    self.results.append(results)
                    t.update()
        self.results = pd.DataFrame(self.results)
    
def process_disease_wrapper(disease):
    return exp.process_disease(disease)


def main(process_dir, overwrite, notify):
    with open(os.path.join(process_dir, "params.json")) as f:
        params = json.load(f)
    assert(params["process"] == "disease_subgraph")
    global exp
    exp = DiseaseSubgraph(process_dir, params["process_params"])
    if exp.is_completed():
        exp.load_results()
    elif exp.run():
        exp.save_results()
    exp.plot_results()
