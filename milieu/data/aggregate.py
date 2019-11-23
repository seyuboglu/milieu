"""Aggregate results from different experiments into one csv."""
import os
import json
import logging

import numpy as np
import pandas as pd

from milieu.data.associations import load_diseases
from milieu.util.util import Process


class Aggregate(Process):
    """
    """
    def __init__(self, dir, params):
        """
        Constructor 
        Args: 
            dir (string) directory of the experiment to be run
        """
        super(Aggregate, self).__init__(dir, params)

        # Set the logger

        self.params = params
        # Unpack parameters
        self.experiments = self.params["experiments"]
        self.groups_columns = self.params["groups_columns"]

        # Log title 
        logging.info("Aggregating Experiments")
        logging.info("Sabri Eyuboglu  -- SNAP Group")
        logging.info("======================================")
        logging.info("Loading Disease Associations...")
        self.diseases = load_diseases(self.params["associations_path"], 
                                      self.params["disease_subset"],
                                      exclude_splits=['none'])
        print(len(self.diseases))
            
    def process_experiment(self, exp_dict):
        """
        Dictionary of experiment info. 
        args:
            exp_dict
        """
        df = pd.read_csv(exp_dict["path"],
                         index_col=0,
                         engine='python')
        df = df.loc[self.diseases.keys()]
        return df[exp_dict["cols"]]
        
    def _run(self):
        """
        Run the aggregation.
        """
        print("Aggregating...")
        self.results = pd.concat([self.process_experiment(exp)
                                  for exp in self.experiments],
                                 axis=1,
                                 keys=[exp["name"]
                                       for exp in self.experiments])
    
    def summarize_results_by_group(self, column):
        """
        Group the results by column. 
        args:
            column  (string) The column used to form the groups
        """
        groups = self.results.groupby(column)
        return groups.describe()

    def save_results(self, summary=True):
        """
        Saves the results to a csv using a pandas Data Fram
        """
        print("Saving Results...")
        self.results.to_csv(os.path.join(self.dir, 'results.csv'))

        if summary:
            summary_df = self.summarize_results()
            summary_df.to_csv(os.path.join(self.dir, 'summary.csv'))
        
        for column in self.groups_columns:
            summary_df = self.summarize_results_by_group(tuple(column))
            summary_df.to_csv(os.path.join(self.dir, 'summary_{}.csv'.format(column[-1])))


def main(process_dir, overwrite, notify):
    with open(os.path.join(process_dir, "params.json")) as f:
        params = json.load(f)
    assert(params["process"] == "aggregate")
    process = Aggregate(process_dir, params["process_params"])
    process.run()
    process.save_results()
