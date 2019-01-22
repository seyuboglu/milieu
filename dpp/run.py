"""
Module for running experiments. 
"""
import sys
import logging
import os
import json

import dpp.experiments.dpp_evaluate as dpp_evaluate
import dpp.experiments.dpp_predict as dpp_predict


import click


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
    