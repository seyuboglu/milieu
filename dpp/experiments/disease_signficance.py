"""Run experiment"""

import argparse
import logging
import os
import datetime
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import truncnorm, rankdata
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from data import load_diseases, load_network
from exp import Experiment
from method.ppi_matrix import load_ppi_matrices
from util import Params, set_logger, prepare_sns, string_to_list, fraction_nonzero

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='experiments/base_model',
                    help="Directory containing params.json")


def sort_by_degree(network):
    """
    Sort and argsort nodes by degree. Note: assumes that node numbers match index.  
    args:
        network (networkx graph)
    return:
        node_sorted_by_deg (ndarray) List of all nodes sorted by degree
        node_to_rank_by_deg (ndarray) Argsort of nodes by degree 
    """
    # get degrees and nodes 
    degrees =  np.array(network.degree())[:, 1]
    nodes_sorted_by_deg = degrees.argsort()
    nodes_ranked_by_deg = rankdata(degrees, method='ordinal') - 1

    return nodes_sorted_by_deg, nodes_ranked_by_deg


def get_pairwise_scores(ppi_matrix, pathway):
    """
    Gets the scores between nodes in a disease pathway.
    args:
        ppi_matrix  (ndarray)
        pathway (ndarray) 
    """
    inter_pathway = ppi_matrix[pathway, :][:, pathway]

    # ignore diagonal
    above_diag = inter_pathway[np.triu_indices(len(pathway))]
    below_diag = inter_pathway[np.tril_indices(len(pathway))]
    pathway_scores = np.concatenate((above_diag, below_diag))
    return pathway_scores


def density(name, ppi_matrix, adj, pathway):
    """
    """
    if name == "CI":
        interactors = np.sum(adj[pathway, :], axis=0, keepdims=False)
        common_interactors = np.where(interactors > 1)[0]
        subgraph = np.concatenate((pathway, common_interactors))
    else:
        subgraph = pathway 
    
    subgraph = adj[subgraph, :][:, subgraph]
    density = subgraph[np.triu_indices(len(subgraph), k=1)].mean()
    return density


def get_loo_scores(ppi_matrix, pathway):
    """
    Gets the scores between nodes in a disease pathway.
    args:
        ppi_matrix  (ndarray)
        pathway (ndarray) 
    """
    loo = LeaveOneOut()
    scores = []
    for train_index, test_index in loo.split(pathway):
        train = pathway[train_index]
        test = pathway[test_index]

        node_score = np.sum(ppi_matrix[test, train])
        scores.append(node_score)
    return np.array(scores)
    

def loo_median(name, ppi_matrix, adj, pathway):
    """
    Computes the median score for each node w.r.t
    the rest of the pathway. 
    args:
        ppi_matrix  (ndarray)
        disease_pathway (ndarray)
    """
    return np.median(get_loo_scores(ppi_matrix, pathway))


def pairwise_mean(name, ppi_matrix, adj, pathway):
    """
    Computes average score between nodes in a pathway. 
    args:
        ppi_matrix  (ndarray)
        disease_pathway (ndarray)
    """
    return np.mean(get_pairwise_scores(ppi_matrix, pathway))


def pairwise_median(name, ppi_matrix, adj, pathway):
    """
    Computes median score between nodes in a pathway. 
    args:
        ppi_matrix  (ndarray)
        disease_pathway (ndarray)
    """
    return np.median(get_pairwise_scores(ppi_matrix, pathway))


def pairwise_nonzero(name, ppi_matrix, adj, pathway):
    """
    Computes fraction nonzero score between nodes in a pathway. 
    args:
        ppi_matrix  (ndarray)
        disease_pathway (ndarray)
    """
    return fraction_nonzero(get_pairwise_scores(ppi_matrix, pathway))


class PermutationTest(Experiment):
    """
    Class for running experiment that assess the significance of a network metric
    between disease proteins. Uses the metghod described in Guney et al. for generating
    random subgraph. 
    """
    def __init__(self, dir):
        """
        Constructor 
        Args: 
            dir (string) directory of the experiment to be run
        """
        super(PermutationTest, self).__init__(dir)

        # Set the logger
        set_logger(os.path.join(self.dir, 'experiment.log'), level=logging.INFO, console=True)

        # unpack parameters 
        self.ppi_matrices = {name: np.load(file) for name, file in self.params.ppi_matrices.items()}
        self.exclude = set(self.params.exclude) if hasattr(self.params, "exclude") else set()

        # Log title 
        logging.info("Metric Significance of Diseases in the PPI Network")
        logging.info("Sabri Eyuboglu  -- SNAP Group")
        logging.info("======================================")
        logging.info("Loading Disease Associations...")
        self.diseases = load_diseases(self.params.diseases_path, 
                                      self.params.disease_subset,
                                      exclude_splits=['none'])

        self.metric_fn = globals()[self.params.metric_fn]

    def get_null_pathways(self, pathway, quantity=1, stdev=25):
        """
        Given a reference pathway, generate quantity 
        """
        null_pathways = [set() for _ in range(quantity)] 
        for node in pathway:
            node_rank = self.nodes_ranked_by_deg[node]
            a = (0 - node_rank) / stdev
            b = (len(self.nodes_sorted_by_deg) - node_rank) / stdev
            rank_dist = truncnorm(a, b, loc=node_rank, scale=stdev)
            for null_pathway in null_pathways:
                while True:
                    sample_rank = int(rank_dist.rvs())
                    sample_node = self.nodes_sorted_by_deg[sample_rank]
                
                    # gaurantee that the same node is not added twice 
                    if sample_node not in null_pathway:
                        null_pathway.add(sample_node)
                        break
        
        return map(np.array, (map(list, null_pathways)))

    def compute_pvalue(self, disease_result, null_results, random_equal=False):
        # if equal, randomly 
        if random_equal:
            equal = np.logical_and(np.isclose(null_results, disease_result),
                                   np.random.choice(a=[False, True], 
                                                    size=len(null_results)))
        else:
            equal = np.isclose(null_results, disease_result)

        return np.logical_or((null_results > disease_result), equal).mean()


    def process_disease(self, disease):
        """
        Generates null model for disease and computes 
        Args:
            disease (Disease) the current disease 
        """
        disease_pathway = disease.to_node_array(self.protein_to_node)
        null_pathways = self.get_null_pathways(disease_pathway, 
                                               self.params.n_random_pathways, 
                                               self.params.sd_sample)
        results = {"disease_id": disease.id,
                   "disease_name": disease.name,
                   "disease_size": len(disease_pathway)}
        for name, ppi_matrix in self.ppi_matrices.items():
            
            disease_result = self.metric_fn(name, ppi_matrix, 
                                            self.ppi_adj, disease_pathway)
            null_results = np.array([self.metric_fn(name, ppi_matrix, 
                                                    self.ppi_adj, null_pathway) 
                                     for null_pathway in null_pathways])

            disease_pvalue = self.compute_pvalue(disease_result, 
                                                 null_results, 
                                                 random_equal=False)
            results.update({"pvalue_" + name: disease_pvalue,
                            self.metric_fn.__name__ + "_" + name: disease_result,
                            "null_" + self.metric_fn.__name__ + "s_" + name: null_results})
            
        return results
    
    def _run(self):
        """
        Run the experiment.
        """
        logging.info("Loading Network...")
        self.ppi_networkx, self.ppi_adj, self.protein_to_node = load_network(
            self.params.ppi_network) 

        logging.info("Loading PPI Matrices...")
        self.ppi_matrices = load_ppi_matrices(self.params.ppi_matrices)

        logging.info("Sorting Nodes by Degree...")
        self.nodes_sorted_by_deg, self.nodes_ranked_by_deg = sort_by_degree(
            self.ppi_networkx)

        logging.info("Running Experiment...")
        self.results = []

        if self.params.n_processes > 1:
            with tqdm(total=len(self.diseases)) as t: 
                p = Pool(self.params.n_processes)
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
        for name in self.ppi_matrices.keys():
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
    
    def plot_disease(self, disease):
        """
        Plots one disease 
        """
        row = self.results.loc[self.results['disease_id'] == disease.id]
        disease_dir = os.path.join(self.dir, 'figures', 'diseases', disease.id)

        if not os.path.exists(disease_dir):
            os.makedirs(disease_dir)

        for name in self.ppi_matrices.keys():
            null_results = row["null_" + self.metric_fn.__name__ + "s_" + name].values[0]

            if type(null_results) == str:
                null_results = string_to_list(null_results, float)
            
            if np.allclose(null_results, 0):
                return

            if self.params.plot_type == "bar":
                sns.distplot(null_results, kde=False, bins=40, 
                             color="grey", label="Random Pathways")
                plt.ylabel("Pathways [count]")
 
            elif self.params.plot_type == "kde":
                sns.kdeplot(null_results, shade=True, kernel="gau", 
                            color="grey", label="Random Pathways")
                plt.ylabel("Pathways [KDE]")
                plt.yticks([])

            elif self.params.plot_type == "bar_kde":
                sns.distplot(null_results, kde=True, bins=20, 
                            color="grey", label="Random Pathways") 
                plt.ylabel("Pathways [count]")

            disease_mean = row[self.metric_fn.__name__ + "_" + name]
            sns.scatterplot(disease_mean, 0, label=disease.name)

            plt.xlabel(self.params.xlabel_disease.format(self.params.labels[name] if hasattr(self.params, "labels") else name))
            sns.despine()

            plt.tight_layout()
            plt.savefig(os.path.join(disease_dir, 
                                     name + "_{}_{}.pdf".format(self.params.metric_fn,
                                                                self.params.plot_type)))
            plt.close()
            plt.clf()
    
    def plot_all_diseases(self):
        """
        Estimates the distribution of z-scores for each metric across all diseases
        then plots the estimated distributions on the same plot. 
        """
        plt.rc("xtick", labelsize=6)

        figures_dir = os.path.join(self.dir, 'figures')
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)

        for name in self.ppi_matrices.keys(): 
            if name in self.exclude:
                continue 
            series = self.results["pvalue_" + name]
            series = np.array(series)
            #sns.barplot(series, label=name)
            if self.params.plot_type == "bar":
                sns.distplot(series, bins=40, kde=False, 
                             hist_kws={'range': (0.0, 0.25)}, 
                             label=self.params.labels[name] 
                                   if hasattr(self.params, "labels") 
                                   else name)
                plt.ylabel("Pathways [count{}]".format(r' $\log_{10}$' 
                                                       if self.params.yscale == "log" 
                                                       else ""))

            elif self.params.plot_type == "kde":
                sns.kdeplot(series, shade=True, kernel="gau", clip=(0, 1), 
                            label=self.params.labels[name] 
                                  if hasattr(self.params, "labels") 
                                  else name)
                plt.ylabel("Pathways [KDE{}]".format(r' $\log_{10}$' 
                                                     if self.params.yscale == "log" 
                                                     else ""))
                plt.yticks([])

            elif self.params.plot_type == "bar_kde":
                sns.distplot(series, bins=40, kde=True, 
                             kde_kws={'clip': (0.0, 1.0)}, label=name)
                plt.ylabel("Pathways [count{}]".format(r' $\log_{10}$' 
                                                       if self.params.yscale == "log" 
                                                       else ""))
            
            elif self.params.plot_type == "":
                pass

        plt.xlabel(self.params.xlabel_all)
        sns.despine()
        plt.xticks(np.arange(0.0, 1.0, 0.05))
        if self.params.plot_type == "kde": 
            plt.yticks()
        plt.legend()
        #plt.tight_layout()
        plt.xlim(xmin=0, xmax=0.25)
        plt.yscale(self.params.yscale)

        time_string = datetime.datetime.now().strftime("%m-%d_%H%M")
        plot_path = os.path.join(figures_dir, 
                                 'pvalue_dist_{}_{}_'.format(self.params.plot_type, 
                                                             self.params.yscale) + time_string + '.pdf')
        plt.savefig(plot_path)
        plt.close()
        plt.clf()

    def plot_results(self):
        """
        Plots the results 
        """
        print("Plotting Results...")
        prepare_sns(sns, self.params)
        self.plot_all_diseases()

        for disease_id in tqdm(self.params.disease_plots):
            self.plot_disease(self.diseases[disease_id])


def process_disease_wrapper(disease):
    return exp.process_disease(disease)


if __name__ == "__main__":
    args = parser.parse_args()
    exp = PermutationTest(args.dir)
    if exp.is_completed():
        exp.load_results()
    elif exp.run():
        exp.save_results()
    exp.plot_results()