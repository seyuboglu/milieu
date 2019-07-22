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

from dpp.util import set_logger, send_email
import dpp.experiments.dpp_evaluate as dpp_evaluate
import dpp.experiments.dpp_predict as dpp_predict
import dpp.experiments.go_enrichment as go_enrichment
import dpp.experiments.protein_significance as protein_significance
import dpp.experiments.disease_significance as disease_significance
import dpp.experiments.disease_subgraph as disease_subgraph
import dpp.experiments.network_robustness as network_robustness
import dpp.data.network_matrices as network_matrices
import dpp.figures.recall_curve as recall_curve
import dpp.figures.recall_comparison as recall_comparison
import dpp.data.aggregate as aggregate 


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
    
