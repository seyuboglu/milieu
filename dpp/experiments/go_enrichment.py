"""Run experiment"""
import logging
import os
import json
import datetime
import time
from multiprocessing import Pool

import numpy as np
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
    
    def run_study(self, proteins):
        """
        """
        results = self.enrichment_study.run_study(proteins)
        significant_terms = [r.goterm.name for r in results if r.p_fdr_bh < 0.05]
        top_k_terms = [r.goterm.name 
                       for r in sorted(results, 
                                       key=lambda x: x.p_fdr_bh)[:self.params["top_k"]]]


        return set(significant_terms), set(top_k_terms)

    def process_disease(self, disease):
        """
        """
        results = {}
        # compute method scores for disease
        disease_proteins = set(self.diseases_dict[disease.id].proteins)
        disease_terms, top_disease_terms = self.run_study(disease_proteins)
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
            
            pred_terms, top_pred_terms = self.run_study(pred_proteins)
            intersection = (disease_terms & pred_terms)
            union = (disease_terms | pred_terms)
            jaccard = 0 if len(union) == 0 else len(intersection) / len(union)

            results["{}_num_significant".format(name)] = len(pred_terms)
            results["{}_top_{}".format(name, self.params['top_k'])] = top_pred_terms
            results["{}_jaccard_sim".format(name)] = jaccard

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
