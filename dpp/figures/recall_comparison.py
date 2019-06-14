import json
import argparse
import logging
import os
import csv
from multiprocessing import Pool

import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
from scipy.stats import rankdata

from dpp.data.network import PPINetwork
from dpp.data.associations import load_diseases
from dpp.figures.figure import Figure
from dpp.util import Params, set_logger, parse_id_rank_pair, prepare_sns

 
class RecallComparison(Figure):
    """ 
    Base class for all disease protein prediction methods.
    """
    def __init__(self, dir, params):
        """ Initialize the 
        Args: 
            dir (string) The directory where the experiment should be run
            params  (dict)
        """
        super().__init__(dir, params)
        self._load_data()
        prepare_sns(sns, self.params)
        logging.info("Recall Comparison")
        logging.info("Sabri Eyuboglu  -- SNAP Group")
        logging.info("======================================")

    def _run(self):
        """
        """
        self._generate()

    def _load_data(self):
        """
        """
        logging.info("Loading Disease Associations...")
        self.diseases_dict = load_diseases(self.params["diseases_path"])
    
        logging.info("Loading Results...")
        method_to_metrics = {}
        for name, exp_dir in self.params["method_exp_dirs"].items():
            metrics = pd.read_csv(os.path.join(exp_dir, "metrics.csv"))
            method_to_metrics[name] = metrics
        self.method_to_metrics = method_to_metrics
        
    def _generate(self):
        """
        """    
        sns.set(font_scale=2, rc={'figure.figsize':(4, 3)})
        prepare_sns(sns)

        for name, metrics in self.method_to_metrics.items():
            if name == self.params["reference"]:
                continue
            ref_name = self.params["reference"]
            ref_metric = self.method_to_metrics[ref_name][self.params["metric"]]
            curr_metric = metrics[self.params["metric"]]
            
            diffs = np.sort(ref_metric - curr_metric)
            
            ref_start = np.where(diffs > 0.0)[0][0]
            method_end = np.where(diffs < 0.0)[-1][-1]
            
            plt.plot(np.arange(ref_start, len(diffs)), diffs[ref_start:],
                 label=ref_name, alpha=0.6)
            plt.fill_between(np.arange(ref_start, len(diffs)), 
                             0, diffs[ref_start:], alpha=0.8)
            plt.plot(np.arange(method_end), diffs[:method_end], 
                     label=name, alpha=0.6)
            plt.plot(np.arange(len(diffs)), np.zeros(len(diffs)), color='k', linewidth=0.8)
            plt.fill_between(np.arange(method_end), 0, diffs[:method_end], alpha=0.8)

            plt.ylabel(f"Difference in {self.params['metric']}") 
            plt.xlabel("Diseases sorted by difference")
            plt.xlim(0, 1850)
            plt.legend()

            sns.despine()
            plt.tight_layout()

            plt.savefig(os.path.join(self.dir, ref_name + "-" + name + ".pdf"))
            plt.clf()
            #break
        

    def save(self): 
        """
        """
        plt.savefig(os.path.join(self.dir, 
                                 f"recall_curve_{self.params['length']}.pdf"))
    
    def show(self):
        """
        """
        plt.show()

def main(process_dir, overwrite, notify):
    with open(os.path.join(process_dir, "params.json")) as f:
        params = json.load(f)
    assert(params["process"] == "recall_curve")
    global exp
    fig = RecallCurve(process_dir, params["process_params"])
    fig.run()
    fig.save()
