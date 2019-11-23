"""Run experiment"""
import sys
import logging
import os
import copy 
import json
from multiprocessing import Pool
import random

import numpy as np
import pandas as pd
import torch

from milieu.paper.experiments.experiment import Experiment
from milieu.paper.experiments.evaluate_method import EvaluateMethod
from milieu.paper.experiments.dpp_predict import DPPPredict
from milieu.paper.experiments.go_enrichment import GOEnrichment
from milieu.paper.experiments.node_significance import NodeSignificance
from milieu.paper.experiments.set_significance import SetSignificance
from milieu.paper.experiments.disease_subgraph import DiseaseSubgraph
from milieu.util.util import set_logger


def run_experiment(experiment_dir):
    with open(os.path.join(experiment_dir, "params.json")) as f:
        params = json.load(f)
    experiment = globals()[params["process"]](experiment_dir, 
                                              params["process_params"])
    experiment.run()
    experiment.save_results()


class NetworkRobustness(Experiment):
    """
    Class for running experiment that compute simple network metrics for
    disease pathways. 
    """
    def __init__(self, dir, params):
        """
        Constructor 
        Args: 
            dir (string) directory of the experiment to be run
        """
        super().__init__(dir, params)

        # set the logger
        set_logger(os.path.join(self.dir, 'experiment.log'), 
                   level=logging.INFO, console=True)        
    
    def _run(self):
        """
        Run the experiments.
        """     
        all_experiments = []
        for config_idx, config in enumerate(self.params["configs"]):
            config_dir = os.path.join(self.dir, f"config_{config_idx}")
            if not os.path.isdir(config_dir):
                os.mkdir(config_dir)
            experiment_params = self.params["experiment_params"]
            experiment_params.update(config)
            params = {
                "process": self.params["experiment_class"],
                "process_params": experiment_params
            }
            for run_idx in range(self.params["num_runs"]):
                run_dir = os.path.join(config_dir, f"run_{run_idx}")
                if not os.path.isdir(run_dir):
                    os.mkdir(run_dir)

                with open(os.path.join(run_dir, "params.json"), 'w') as f:
                    json.dump(params, f)
                all_experiments.append(run_dir)
                
        for idx, experiment_dir in enumerate(all_experiments):
            # set seeds
            seed = idx
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            run_experiment(experiment_dir)
    

def main(process_dir, overwrite, notify):
    """
    """
    with open(os.path.join(process_dir, "params.json")) as f:
        params = json.load(f)
    assert(params["process"] == "network_robustness")
    exp = NetworkRobustness(process_dir, params["process_params"])
    exp.run()
