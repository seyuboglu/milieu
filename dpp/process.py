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