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

import pandas 
import click

from dpp.util import set_logger, send_email
import dpp.experiments.dpp_evaluate as dpp_evaluate
import dpp.experiments.dpp_predict as dpp_predict
import dpp.experiments.go_enrichment as go_enrichment
import dpp.experiments.protein_significance as protein_significance
import dpp.experiments.disease_significance as disease_significance
import dpp.data.network_matrices as network_matrices
import dpp.figures.recall_curve  as recall_curve


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
def main(process_dir, overwrite, notify):
    """
    """
    with open(os.path.join(process_dir, "params.json")) as f:
        params = json.load(f)
    process = globals()[params["process"]].main(process_dir, overwrite, notify)

if __name__ == "__main__":
    sys.exit(main())
    