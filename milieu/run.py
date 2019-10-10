"""
Module for running experiments. 
"""
import sys
import logging
import os
import json
import smtplib
import socket
import traceback
import random
from shutil import copyfile

import pandas as pd
import numpy as np
import torch 
import click

from milieu.util.util import set_logger, send_email
import milieu.experiments.dpp_evaluate as dpp_evaluate
import milieu.experiments.dpp_predict as dpp_predict
import milieu.experiments.go_enrichment as go_enrichment
import milieu.experiments.protein_significance as protein_significance
import milieu.experiments.disease_significance as disease_significance
import milieu.experiments.disease_subgraph as disease_subgraph
import milieu.experiments.network_robustness as network_robustness
import milieu.data.network_matrices as network_matrices
import milieu.figures.recall_curve as recall_curve
import milieu.figures.robustness_curve as robustness_curve
import milieu.figures.recall_comparison as recall_comparison
import milieu.data.aggregate as aggregate 


class Process(object):
    """ 
    Base class for all disease protein prediction processes.
    """
    def __init__(self, dir, params):
        """ Initialize the 
        Args: 
            dir (string) The directory where the experiment should be run
            params  (dict)
        """
        self.dir = dir
        # ensure dir exists
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
            with open(os.path.join(self.dir, "params.json"), 'w') as f: 
                json.dump(params, f, indent=4)
        
        self.params = params
        set_logger(os.path.join(self.dir, 'process.log'), 
                   level=logging.INFO, 
                   console=True)
    
    def is_completed(self):
        """
        Checks if the experiment has already been run. 
        """
        return os.path.isfile(os.path.join(self.dir, 'results.csv'))

    def _run(self): 
        raise NotImplementedError
    
    def run(self, overwrite=False):
        if os.path.isfile(os.path.join(self.dir, 'results.csv')) and not overwrite:
            print("Experiment already run.")
            return False 

        if hasattr(self.params, "notify") and self.params.notify:
            try:
                self._run()
            except:
                tb = traceback.format_exc()
                self.notify_user(error=tb)
                return False 
            else:
                self.notify_user()
                return True 
        self._run()
        return True
        
    def notify_user(self, error=None):
        # read params 
        with open(os.path.join(self.dir, 'params.json'), "r") as file:
            params_string = file.readlines()
        if error is None:
            subject = "Experiment Completed: " + self.dir
            message = ("Yo!\n",
                       "Good news, your experiment just finished.",
                       "You were running the experiment on: {}".format(
                           socket.gethostname()),
                       "---------------------------------------------",
                       "See the results here: {}".format(self.dir),
                       "---------------------------------------------", 
                       "The parameters you fed to this experiment were: {}".format(
                           params_string),
                       "---------------------------------------------", 
                       "Thanks!")
        else: 
            subject = "Experiment Error: " + self.dir
            message = ("Uh Oh!\n",
                       "Your experiment encountered an error.",
                       "You were running the experiment found at: {}".format(self.dir),
                       "You were running the experiment on: {}".format(
                           socket.gethostname()),
                       "---------------------------------------------",
                       "Check out the error message: \n{}".format(error),
                       "---------------------------------------------", 
                       "The parameters you fed to this experiment were: {}".format(
                           params_string),
                       "---------------------------------------------", 
                       "Thanks!")

        message = "\n".join(message)
        send_email(subject, message)

    def __call__(self): 
        return self.run()
    
    def summarize_results(self):
        """
        Returns the summary of the dataframe 
        return:
            summary_df (DataFrame) 
        """
        return self.results.describe()


@click.command()
@click.argument(
    "process_dir",
    type=str,
)
@click.option(
    "--overwrite",
    type=bool,
    default=False
)
@click.option(
    "--notify",
    type=bool,
    default=True
)
@click.option(
    "--num_runs",
    type=int,
    default=1
)
def main(process_dir, overwrite, notify, num_runs):
    """
    """
    if num_runs == 1:
        with open(os.path.join(process_dir, "params.json")) as f:
            params = json.load(f)
        process = globals()[params["process"]].main(process_dir, overwrite, notify)

    elif num_runs > 1:
        for idx in range(num_runs):
            run_dir = os.path.join(process_dir, f"run_{idx}")
            os.mkdir(run_dir)
            params_path = os.path.join(process_dir, "params.json")
            copyfile(params_path, os.path.join(run_dir, "params.json"))
            with open(params_path) as f:
                params = json.load(f)
            # set seeds
            seed = idx
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            process = globals()[params["process"]].main(run_dir, overwrite, notify)


if __name__ == "__main__":
    sys.exit(main())
    
