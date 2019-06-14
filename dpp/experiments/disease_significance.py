"""Run experiment"""

import argparse
import logging
import os
import datetime
import json
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import truncnorm, rankdata
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from dpp.data.associations import load_diseases
from dpp.data.network import PPINetwork
from dpp.data.network_matrices import load_network_matrices
from dpp.experiments.experiment import Experiment
from dpp.util import (Params, set_logger, prepare_sns, string_to_list, 
                      compute_pvalue, build_degree_buckets, list_to_string)


def loo_iter(disease_pathway):
        """
        """
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(disease_pathway):
            train_nodes = disease_pathway[train_index]
            test_node = disease_pathway[test_index][0]
            yield (train_nodes, test_node)    


class DiseaseSignificance(Experiment):
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

    def get_null_pathways(self, pathway, quantity=1):
        """
        Given a reference pathway, generate quantity 
        """
        null_pathways = np.zeros((quantity, len(pathway)), dtype=int)
        null_pathways 
        for i, node in enumerate(pathway):
            degree = self.network.nx.degree[node]
            null_nodes = np.random.choice(self.degree_to_bucket[degree], 
                                          size=quantity, 
                                          replace=True)
            null_pathways[:, i] = null_nodes
        
        return [null_pathways[i, :] for i in range(quantity)]
    
    def mean_frac_direct_interactions(self, disease_pathway):
        """
        """
        fracs = []
        for train_nodes, test_node in loo_iter(disease_pathway):
            frac = (np.count_nonzero(np.dot(self.network.adj_matrix[test_node, :], 
                                     self.network.adj_matrix[train_nodes, :].T)) 
                    / len(train_nodes))
            fracs.append(frac)
        return np.mean(fracs)
    
    def mean_common_interactor_score(self, disease_pathway):
        """
        """
        ci_scores = []
        for train_nodes, test_node in loo_iter(disease_pathway):
            ci_score = np.sum(self.ppi_matrices["ci"][test_node, train_nodes])
            ci_scores.append(ci_score)
        return np.mean(ci_scores)
    
    def process_disease(self, disease):
        """
        Generates null model for disease and computes 
        Args:
            disease (Disease) the current disease 
        """
        disease_pathway = disease.to_node_array(self.network)
        null_pathways = self.get_null_pathways(disease_pathway, 
                                               self.params["n_random_pathways"])
        results = {"disease_id": disease.id,
                   "disease_name": disease.name,
                   "disease_size": len(disease_pathway)}
        for metric_fn in self.params["metric_fns"]:
            disease_result = getattr(self, metric_fn)(disease_pathway)
            null_results = np.array([getattr(self, metric_fn)(null_pathway) 
                                     for null_pathway in null_pathways])

            disease_pvalue = compute_pvalue(disease_result, null_results)
            results.update({"pvalue_" + metric_fn: disease_pvalue,
                            "disease_" + metric_fn: disease_result,
                            "null_" + metric_fn: list_to_string(null_results)})
            
        return results
    
    def _run(self):
        """
        Run the experiment.
        """
        logging.info("Loading Network...")
        self.network = PPINetwork(self.params["ppi_network"]) 

        logging.info("Loading PPI Matrices...")
        self.ppi_matrices = load_network_matrices(self.params["ppi_matrices"], 
                                                  self.network)

        logging.info("Building Degree Buckets...")
        self.degree_to_bucket = build_degree_buckets(self.network,
                                                     min_len=self.params["min_bucket_len"])

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
    
    def summarize_results(self):
        """
        Creates a dataframe summarizing the results across
        diseases. 
        return:
            summary_df (DataFrame)
        """
        summary_df =  self.results.describe()

        frac_significant = {} 
        for name in self.params["metric_fns"]:
            col_name = "pvalue_" + name
            col = self.results[col_name]
            frac_significant[col_name] = np.mean(np.array(col) < 0.05)
        summary_df = summary_df.append(pd.Series(frac_significant, name="< 0.05"))

        return summary_df
    
    def save_results(self, summary=True):
        """
        Saves the results to a csv using a pandas Data Fram
        """
        print("Saving Results...")
        self.results.to_csv(os.path.join(self.dir, 'results.csv'), index=False)

        if summary:
            summary_df = self.summarize_results()
            summary_df.to_csv(os.path.join(self.dir, 'summary.csv'))
    
    def load_results(self):
        """
        Loads the results from a csv to a pandas Data Frame.
        """
        print("Loading Results...")
        self.results = pd.read_csv(os.path.join(self.dir, 'results.csv'))
    
    def plot_disease(self, name, disease_ids,
                     metrics, plot_type="bar", 
                     xlabel="", ylabel="",
                     yscale="linear", bins=100,
                     xmin=0.0, xmax=1.0,
                     null_color='lightgrey'):
        """
        Plots one disease 
        """
        sns.set(font_scale=2, rc={'figure.figsize':(4, 2)})
        prepare_sns(sns, self.params)
        for disease_id in disease_ids:
            print(disease_id)
            row = self.results.loc[self.results['disease_id'] == disease_id]
            disease_dir = os.path.join(self.dir, 'figures', 'diseases', disease_id)

            if not os.path.exists(disease_dir):
                os.makedirs(disease_dir)

            for metric in metrics:
                disease_mean = row["disease_" + metric]
                null_results = row["null_" + metric].values[0]

                if type(null_results) == str:
                    null_results = string_to_list(null_results, float)

                xmin = min((disease_mean.item(), min(null_results)))
                xmax = max((disease_mean.item(),))
                delta = np.abs(xmax - xmin) * 0.1
                xmin -= delta
                xmax += delta

                if plot_type == "bar":
                    sns.distplot(null_results, kde=False, bins=bins, 
                                 hist_kws={'range': (xmin,xmax)},
                                 color=null_color, label="random pathways")
                    plt.ylabel(ylabel)
    
                elif plot_type == "kde":
                    sns.kdeplot(null_results, shade=True, kernel="gau", 
                                color=null_color, label="random Pathways")
                    plt.ylabel(ylabel)
                    plt.yticks([])

                elif plot_type == "bar_kde":
                    ax = plt.gca()
                    fig2, ax2 = plt.subplots()
                    sns.distplot(null_results, hist=False, kde=True,
                                 color=null_color, label="Random Pathways")
                    ax = sns.distplot(null_results, ax=ax2, kde=False, bins=bins, 
                                      hist_kws={'range': (xmin, xmax)},
                                      color=null_color, label="Random Pathways", norm_hist=False)
                    ax.yaxis = ax2.yaxis
                    plt.ylabel(ylabel)
                    plt.ylabel(ylabel)

                disease = self.diseases[disease_id]
                sns.scatterplot(disease_mean, 0)
                
                xlabel = r"Mean CI score" # , $\frac{1}{|S|} \sum_{u \in S} CI(u, S \backslash \{u\})$
                plt.xlabel(xlabel)
                sns.despine()

                plt.tight_layout()
                plt.savefig(os.path.join(disease_dir, 
                                         f"{disease_id}_simple.pdf"))
                plt.show()
                plt.close()
                plt.clf()
    
    def plot_full_distribution(self, name, metrics, plot_type="bar", 
                               xlabel="", ylabel="",
                               yscale="linear", bins=100,
                               xmin=0.0, xmax=1.0):
        """
        Estimates the distribution of z-scores for each metric across all diseases
        then plots the estimated distributions on the same plot. 
        """
        plt.rc("xtick", labelsize=6)

        figures_dir = os.path.join(self.dir, 'figures')
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)

        for metric in metrics: 
            series = self.results["pvalue_" + metric]
            series = np.array(series)
            if plot_type == "bar":
                sns.distplot(series, bins=40, kde=False, 
                             hist_kws={'range': (0.0, 0.25)}, 
                             label=metric)
                plt.ylabel("Pathways [count{}]".format(r' $\log_{10}$' 
                                                       if yscale == "log" 
                                                       else ""))

            elif plot_type == "kde":
                sns.kdeplot(series, shade=True, kernel="gau", clip=(0, 1), 
                            label=metric)
                plt.ylabel("Pathways [KDE{}]".format(r' $\log_{10}$' 
                                                     if yscale == "log" 
                                                     else ""))
                plt.yticks([])

            elif plot_type == "bar_kde":
                sns.distplot(series, bins=40, kde=True, 
                             kde_kws={'clip': (0.0, 1.0)}, label=metric)
                plt.ylabel("Pathways [count{}]".format(r' $\log_{10}$' 
                                                       if self.params["yscale"] == "log" 
                                                       else ""))
            
            elif plot_type == "":
                pass

        plt.xlabel(xlabel)
        sns.despine()
        plt.xticks(np.arange(0.0, 1.0, 0.05))
        if plot_type == "kde": 
            plt.yticks()
        print("hello")
        #plt.legend()
        #plt.tight_layout()
        plt.xlim(left=0, right=0.25)
        plt.yscale(yscale)

        time_string = datetime.datetime.now().strftime("%m-%d_%H%M")
        plot_path = os.path.join(figures_dir, 
                                 'pvalue_dist_{}_{}_'.format(plot_type, 
                                                             yscale) + time_string + '.pdf')
        plt.savefig(plot_path)
        plt.close()
        plt.clf()

    def plot_results(self):
        """
        Plots the results 
        """
        print("Plotting Results...")
        prepare_sns(sns, self.params)
        self.figures_dir = os.path.join(self.dir, 'figures')
        if not os.path.exists(self.figures_dir):
            os.makedirs(self.figures_dir)
        
        for plot_name, params in self.params["plots_to_params"].items():
            plot_fn = params["plot_fn"]
            del params["plot_fn"]
            getattr(self, plot_fn)(name=plot_name, **params)


def process_disease_wrapper(disease):
    return exp.process_disease(disease)


def main(process_dir, overwrite, notify):
    with open(os.path.join(process_dir, "params.json")) as f:
        params = json.load(f)
    assert(params["process"] == "disease_significance")
    global exp
    exp = DiseaseSignificance(process_dir, params["process_params"])
    if exp.is_completed():
        exp.load_results()
    elif exp.run():
        exp.save_results()
    exp.plot_results()
