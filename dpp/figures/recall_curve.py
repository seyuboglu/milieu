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
from scipy.stats import rankdata

from dpp.data.network import PPINetwork
from dpp.data.associations import load_diseases
from dpp.figures.figure import Figure
from dpp.util import Params, set_logger, parse_id_rank_pair, prepare_sns

 
class RecallCurve(Figure):
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
        count = 0
        recall_curves = {}
        for name, method_exp_dir in self.params["method_exp_dirs"].items():
            print(name)
            if(self.params["by_disease"]):
                recall_curve_sum = np.zeros(self.params["length"])
                with open(os.path.join(method_exp_dir, 'ranks.csv'), 'r') as ranks_file:
                    rank_reader = csv.reader(ranks_file)
                    for i, row in enumerate(rank_reader):
                        if i == 0: 
                            continue
                        if (("associations_threshold" in self.params) and 
                            self.params["associations_threshold"] > len(row) - 2):
                            continue
                        if (("splits" in self.params) and 
                            self.diseases_dict[row[0]].split not in self.params["splits"]):
                            continue 
                        if self.diseases_dict[row[0]].split == "none":
                            continue 
                        count += 1
                        ranks = [parse_id_rank_pair(rank_str)[1] for rank_str in row[2:]]
                        ranks = np.array(ranks).astype(int)
                        rank_bin_count = np.bincount(ranks)
                        recall_curve = 1.0 * np.cumsum(rank_bin_count) / len(ranks)
                        if len(recall_curve) < self.params["length"]:
                            recall_curve = np.pad(recall_curve, 
                                                (0, self.params["length"] - len(recall_curve)), 
                                                'edge')
                        recall_curve_sum += recall_curve[:self.params["length"]]
                recall_curve = (recall_curve_sum / (count))
                for threshold in self.params["thresholds"]:
                    print(f"recall-at-{threshold}: {recall_curve[threshold]}")
                recall_curves[name] = recall_curve
                plt.plot(recall_curve, label=name, linewidth=2.0 if name == self.params["primary"] else 1.3 )
                print(count)
                count = 0
            else: 
                ranks = []
                with open(os.path.join(method_exp_dir, 'ranks.csv'), 'r') as ranks_file:
                    rank_reader = csv.reader(ranks_file)
                    for i, row in enumerate(rank_reader):
                        if i == 0: continue
                        ranks.extend(map(float, row[2:]))
                ranks = np.array(ranks).astype(int)
                rank_bin_count = np.bincount(ranks)
                recall_curve = 1.0 * (np.cumsum(rank_bin_count) / len(ranks))
                plt.plot(recall_curve[:self.params["length"]], label=name)
            
        # plot percent differences
        for k in self.params["thresholds"]:
            recalls_at_k = [] 
            for name, recall_curve in recall_curves.items():
                recalls_at_k.append(recall_curve[k])
            recalls_at_k.sort(reverse=True)
            plt.plot([k, k], [recalls_at_k[0] - self.params["offset"], recalls_at_k[1] + self.params["offset"]], linestyle='--', color='green', alpha=0.5)
            percent_increase = 1.0 * round(100 * (recalls_at_k[0] - recalls_at_k[1]) / recalls_at_k[1], 1)
            plt.text(x=k + (1.0/100)*self.params["length"], y=recalls_at_k[1] + (recalls_at_k[0] - recalls_at_k[1]) / 2 - 0.001, 
                     s='+' + str(percent_increase) + '%',
                    fontsize=9, weight='bold', alpha=.75, color='green')
        
        if(self.params["title"]):
            plt.title("Recall-at-K (%) across methods")
    
        sns.despine()

        plt.ylabel(r"Recall-at-$k$")
        plt.xlabel(r"Threshold [$k$]")

        plt.tight_layout()

        plt.legend(loc='upper left')
        

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
