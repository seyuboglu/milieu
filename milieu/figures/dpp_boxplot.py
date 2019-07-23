import json
import argparse
import logging
import os
import csv
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import rankdata

from milieu.data.network import PPINetwork
from milieu.data.associations import load_diseases
from milieu.figures.figure import Figure
from milieu.util import Params, set_logger, parse_id_rank_pair, prepare_sns

 
class DPPBoxplot(Figure):
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
        logging.info("Recall Curve")
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

    def _generate(self):
        """
        """
        sns.set_palette("Reds")

        count = 0
        metrics = {}
        for name, method_exp_dir in self.params["method_exp_dirs"].items():
            logging.info(name)
            metrics_df = pd.read_csv(os.path.join(method_exp_dir, "metrics.csv"), index_col=0)
            metric = metrics_df[self.params["metric"]]
            metrics[name] = metric
        metrics_df = pd.DataFrame.from_dict(metrics)
        sns.barplot(data=metrics_df)

        sns.despine()

        plt.ylabel(r"Mean Recall-at-25")
        plt.xlabel(r"Fraction of edges removed")

        plt.tight_layout()

        plt.savefig(os.path.join(self.dir, 
                                 f"network_robustness.pdf"))
    
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
