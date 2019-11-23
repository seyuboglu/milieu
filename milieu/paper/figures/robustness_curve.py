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

from milieu.data.network import Network
from milieu.data.associations import load_diseases
from milieu.paper.figures.figure import Figure
from milieu.util.util import set_logger, parse_id_rank_pair, prepare_sns

 
class RobustnessCurve(Figure):
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
        self.diseases_dict = load_diseases(self.params["associations_path"])

    def _generate(self):
        """
        """
        plt.figure(figsize=(7,3))

        count = 0
        metrics = []
        for name, method_exp_dir in self.params["method_exp_dirs"].items():
            if os.path.isdir(os.path.join(method_exp_dir, 'run_0')):
                for dir_name in os.listdir(method_exp_dir):
                    if dir_name[:3] != "run":
                        continue
                    curr_df = pd.read_csv(os.path.join(method_exp_dir, dir_name, "metrics.csv"), 
                                          index_col=0)
                    mean_metric = curr_df[self.params["metric"]].mean()
                    metrics.append({
                        "fraction": float(name),
                        f"mean_{self.params['metric']}": mean_metric
                    })                

            else: 
                logging.info(name)
                curr_df = pd.read_csv(os.path.join(method_exp_dir, "metrics.csv"), 
                                      index_col=0)
                mean_metric = curr_df[self.params["metric"]].mean()
                metrics.append({
                    "fraction": float(name),
                    f"mean_{self.params['metric']}": mean_metric
                })          

        metrics_df = pd.DataFrame.from_dict(metrics)
        sns.lineplot(x="fraction", y=f"mean_{self.params['metric']}", data=metrics_df,
                     markers=True)
        
        for name, method_exp_dir in self.params["comparison_exp_dirs"].items():
            curr_df = pd.read_csv(os.path.join(method_exp_dir, "metrics.csv"), 
                                  index_col=0)
            mean_metric = curr_df[self.params["metric"]].mean()
            sns.lineplot(x=np.linspace(0, 0.5), y=[mean_metric] * 50,
                         dashes=True, label=name, )


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
