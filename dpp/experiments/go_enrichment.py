"""Run experiment"""
import logging
import os
import json
import datetime
import time
import pickle
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from scipy.stats import spearmanr
import pandas as pd
from tqdm import tqdm
from goatools.obo_parser import GODag
from goatools.associations import read_ncbi_gene2go
from goatools.go_enrichment import GOEnrichmentStudy


from dpp.data.associations import load_diseases
from dpp.data.network import PPINetwork
from dpp.experiments.experiment import Experiment
from dpp.util import Params, set_logger


class GOEnrichment(Experiment):
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
        set_logger(os.path.join(self.dir, 'experiment.log'), 
                   level=logging.INFO, console=True)

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
        
        logging.info("Loading enrichment study...")
        obodag = GODag(self.params["go_path"])
        geneid2go = read_ncbi_gene2go(self.params["gene_to_go_path"], taxids=[9606])
        self.enrichment_study = GOEnrichmentStudy(self.network.get_proteins(),
                                                  geneid2go,
                                                  obodag,
                                                  log=None,
                                                  **self.params["enrichment_params"])

        logging.info("Loading predictions...")
        self.method_to_preds = {name: pd.read_csv(os.path.join(preds, "predictions.csv"), 
                                                  index_col=0) 
                                for name, preds in self.params["method_to_preds"].items()}
        
        enrichment_results_path = os.path.join(self.dir, "enrichment_results.pkl")
        if os.path.exists(enrichment_results_path):
            logging.info("Loading enrichment results...")
            with open(enrichment_results_path, 'rb') as f:
                self.enrichment_results = pickle.load(f)
        else:
            self.enrichment_results = defaultdict(dict)
        
    def run_study(self, proteins):
        """
        """
        results = self.enrichment_study.run_study(proteins)
        term_to_pval = {r.goterm.name: r.p_fdr_bh for r in results}

        return term_to_pval
    
    def compute_spearman_correlation(self, a_term_to_pval, b_term_to_pval):
        """
        """
        terms = list(a_term_to_pval.keys())
        sp_corr, sp_pval = spearmanr([a_term_to_pval[term] for term in terms],
                                     [b_term_to_pval[term] for term in terms])
        return sp_corr, sp_pval

    def process_disease(self, disease):
        """
        """
        results = {}
        # compute method scores for disease
        disease_proteins = set(self.diseases_dict[disease.id].proteins)

        if disease.id in self.enrichment_results["disease"]:
            disease_term_to_pval = self.enrichment_results["disease"][disease.id]
        else:
            disease_term_to_pval = self.run_study(disease_proteins)
            self.enrichment_results["disease"][disease.id] = disease_term_to_pval

        disease_terms = set([term for term, pval 
                             in disease_term_to_pval.items() if pval < 0.05])
        top_disease_terms = set([term for term, _
                                 in sorted(disease_term_to_pval.items(), 
                                           key=lambda x: x[1])[:self.params["top_k"]]])
        
        results = {"disease_name": disease.name, 
                   "disease_num_significant": len(disease_terms),
                   "disease_top_{}".format(self.params['top_k']): top_disease_terms}

        # number of predictions to be made 
        num_preds = (len(disease_proteins) 
                     if self.params["num_preds"] == -1 
                     else self.params["num_preds"])
        
        for name, preds in self.method_to_preds.items():
            pred_proteins = set(map(int, preds.loc[disease.id]
                                              .sort_values(ascending=False)
                                              .index[:num_preds]))

            if disease.id in self.enrichment_results[name]:
                pred_term_to_pval = self.enrichment_results[name][disease.id]
            else: 
                pred_term_to_pval = self.run_study(pred_proteins)
                self.enrichment_results[name][disease.id] = pred_term_to_pval

            pred_terms = set([term for term, pval 
                              in pred_term_to_pval.items() if pval < 0.05])
            top_pred_terms = set([term for term, _
                                  in sorted(pred_term_to_pval.items(), 
                                            key=lambda x: x[1])[:self.params["top_k"]]])

            jaccard = (len(disease_terms & pred_terms) / len(disease_terms | pred_terms) 
                       if len(disease_terms | pred_terms) != 0 else 0)
            sp_corr, sp_pval = self.compute_spearman_correlation(disease_term_to_pval,
                                                                 pred_term_to_pval)

            results[f"{name}_num_significant"] = len(pred_terms)
            results[f"{name}_top_{self.params['top_k']}"] = top_pred_terms
            results[f"{name}_jaccard_sim"] = jaccard
            results[f"{name}_sp_corr"] = sp_corr
            results[f"{name}_sp_pval"] = sp_pval

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
                for disease, result in p.imap(process_disease_wrapper, diseases):
                    results.append(result)
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
        self.results.to_csv(os.path.join(self.dir, 'results.csv'))

        if self.params["save_enrichment_results"]:
            with open(os.path.join(self.dir,'enrichment_results.pkl'), 'wb') as f:
                pickle.dump(self.enrichment_results, f)
            
    def load_results(self):
        """
        Loads the results from a csv to a pandas Data Frame.
        """
        print("Loading Results...")
        self.results = pd.read_csv(os.path.join(self.dir, 'results.csv'))
    

def process_disease_wrapper(disease):
    return exp.process_disease(disease)


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
