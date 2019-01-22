"""Run experiment"""
import logging
import os
import csv
import json
from multiprocessing import Pool

import numpy as np
from sklearn.model_selection import KFold
import networkx as nx
import scipy.sparse as sp
import scipy.stats as stats
from tqdm import tqdm
import click

from dpp.experiments.experiment import Experiment
from dpp.methods.random_walk import RandomWalk
from dpp.data.associations import load_diseases
from dpp.data.network import PPINetwork
from dpp.metrics import positive_rankings, recall_at, recall, auroc, average_precision
from dpp.output import ExperimentResults
from dpp.util import Params, set_logger, parse_id_rank_pair


class DPPEvaluate(Experiment):
    """
    Class for the disease protein prediction experiment
    """
    def __init__(self, dir, params):
        """ Initialize the disease protein prediction experiment 
        Args: 
            dir (string) The directory where the experiment should be run
        """
        super().__init__(dir, params)

        # set the logger
        set_logger(os.path.join(dir, 'experiment.log'), level=logging.INFO, console=True)

        # log Title 
        logging.info("Disease Protein Prediction in the PPI Network")
        logging.info("Sabri Eyuboglu  -- SNAP Group")
        logging.info("======================================")

        # load data from params file
        logging.info("Loading PPI Network...")
        self.network = PPINetwork(self.params["ppi_network"])
        logging.info("Loading Disease Associations...")
        self.diseases_dict = load_diseases(self.params["diseases_path"], 
                                           self.params["disease_subset"], 
                                           exclude_splits=['none'])

        # load method
        self.method = globals()[self.params["method_class"]](self.network,
                                                          self.diseases_dict,
                                                          self.params["method_params"])

  
    def _run(self):
        """ Run the disease protein prediction experiment
        Args: 
            dir (string) The directory where the experiment should be run
        """
        logging.info("Running Experiment...")
        disease_to_metrics, disease_to_ranks = {}, {}
        diseases = list(self.diseases_dict.values())
        diseases.sort(key=lambda x: x.split)
        if self.params["n_processes"] > 1: 
            p = Pool(self.params["n_processes"])
            with tqdm(total=len(self.diseases_dict)) as t:
                for n_finished, (disease, metrics, ranks) in enumerate(p.imap(run_dpp_wrapper, diseases), 1):
                    if metrics != None or ranks != None:
                        disease_to_ranks[disease] = ranks 
                        disease_to_metrics[disease] = metrics
                        t.set_postfix(str="{} Recall-at-100: {:.2f}%".format(disease.id, 100 * metrics["Recall-at-100"]))
                    else:
                        t.set_postfix(str="{} Not Recorded".format(disease.id))
                    t.update()
                
        else: 
            with tqdm(total=len(self.diseases_dict)) as t:
                for n_finished, disease in enumerate(diseases): 
                    disease, metrics, ranks = self.run_dpp(disease)
                    if metrics != None or ranks != None:
                        disease_to_metrics[disease] = metrics
                        disease_to_ranks[disease] = ranks 
                        t.set_postfix(str="{} Recall-at-100: {:.2f}%".format(disease.id, 100 * metrics["Recall-at-100"]))
                    else:
                        t.set_postfix(str="{} Not Recorded".format(disease.id))
                    t.update()
     
        self.results = {"metrics": disease_to_metrics,
                        "ranks": disease_to_ranks}
   
    def compute_node_scores(self, train_nodes, disease):
        """ Get score 
        Args:
            disease: (Disease) A disease object
        """
        scores = self.method.compute_scores(train_nodes, disease)
        self.method 
        return scores

    def run_dpp(self, disease):
        """ Perform k-fold cross validation on disease protein prediction on disease
        Args:
            disease: (Disease) A disease object
        """
        disease_nodes = disease.to_node_array(self.network)
        # Ensure that there are at least 2 proteins
        if disease_nodes.size <= 1:
            return disease, None, None 
        labels = np.zeros((len(self.network), 1))
        labels[disease_nodes, 0] = 1 
        metrics = {}

        # Perform k-fold cross validation
        n_folds = (disease_nodes.size 
                   if (self.params["n_folds"] < 0 or 
                       self.params["n_folds"] > len(disease_nodes))
                   else self.params["n_folds"])
        kf = KFold(n_splits=n_folds, shuffle=False)

        for train_indices, test_indices in kf.split(disease_nodes):
            train_nodes = disease_nodes[train_indices]
            val_nodes = disease_nodes[test_indices]
            
            # compute node scores 
            scores = self.compute_node_scores(train_nodes, disease)

            # compute the metrics of target node
            compute_metrics(metrics, labels, scores, train_nodes, val_nodes)

        avg_metrics = {name: np.mean(values) for name, values in metrics.items()}
        proteins = self.network.get_proteins(metrics["Nodes"])
        ranks = metrics["Ranks"]
        proteins_to_ranks = {protein: ranks for protein, ranks in zip(proteins, ranks)}
        return disease, avg_metrics, proteins_to_ranks 

    def save_results(self):
        write_metrics(self.dir, self.results["metrics"])
        write_ranks(self.dir, self.results["ranks"])


def compute_metrics(metrics, labels, scores, train_nodes, test_nodes):
    """Synthesize the metrics for one disease. 
    Args: 
        metrics: (dictionary) 
        labels: (ndarray) binary array indicating in-disease nodes
        scores: (ndarray) scores assigned to each node
        train_node: (ndarray) array of train nodes
        test_nodes: (ndarray) array of test nodes
    """
    for k in [100, 25]: 
        metrics.setdefault("Recall-at-{}".format(k), []).append(recall_at(labels, 
                                                                          scores, 
                                                                          k, 
                                                                          train_nodes))
    metrics.setdefault("AUROC", []).append(auroc(labels, scores, train_nodes))
    metrics.setdefault("Mean Average Precision", []).append(average_precision(labels, 
                                                                              scores, 
                                                                              train_nodes))
    metrics.setdefault("Ranks", []).extend(positive_rankings(labels, scores, train_nodes))
    metrics.setdefault("Nodes", []).extend(test_nodes)


def run_dpp_wrapper(disease):
    return exp.run_dpp(disease)


def write_metrics(directory, disease_to_metrics):
    """Synthesize the metrics for one disease. 
    Args: 
        directory: (string) directory to save results
        disease_to_metrics: (dict)
    """
    # Output metrics to csv
    output_results = ExperimentResults()
    for disease, metrics in disease_to_metrics.items():  
        output_results.add_disease_row(disease.id, disease.name)
        output_results.add_data_row_multiple(disease.id, metrics)
    output_results.add_statistics()
    output_results.output_to_csv(os.path.join(directory, 'metrics.csv'))


def write_ranks(directory, disease_to_ranks):
    """Write out the ranks for the proteins for one . 
    Args: 
        directory: (string) directory to save results
        disease_to_ranks: (dict)
    """
    # Output metrics to csv
    with open(os.path.join(directory, 'ranks.csv'), 'w') as file:
        ranks_writer = csv.writer(file)
        ranks_writer.writerow(['Disease ID', 'Disease Name', 'Protein Ranks', 'Protein Ids'])
        for curr_disease, curr_ranks in disease_to_ranks.items():
            curr_ranks = [str(protein) + "=" + str(rank) for protein, rank in curr_ranks.items()]
            ranks_writer.writerow([curr_disease.id, curr_disease.name] + curr_ranks)


def load_ranks(directory):
    """Load ranks from a rankings file output for one method. 
    """
    disease_to_ranks = {}
    with open(os.path.join(directory, 'ranks.csv'),'r') as file: 
        ranks_reader = csv.reader(file)
        for i, row in enumerate(ranks_reader):
            if (i == 0): 
                continue 
            disease_id = row[0]
            protein_to_ranks = {id: rank for id, rank in map(parse_id_rank_pair, row[2:])}
            disease_to_ranks[disease_id] = protein_to_ranks
        
    return disease_to_ranks


def main(process_dir, overwrite, notify):
    """
    """
    with open(os.path.join(process_dir, "params.json")) as f:
        params = json.load(f)
    assert(params["process"] == "dpp_evaluate")
    exp = DPPEvaluate(process_dir, params["process_params"])
    if exp.is_completed():
        exp.load_results()
    elif exp.run():
        exp.save_results()
