"""
Provides base class for all experiments 
"""
import os

import pandas 

from dpp.process import Process
 

class Experiment(Process):
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
    
    def summarize_results(self):
        """
        Returns the summary of the dataframe 
        return:
            summary_df (DataFrame) 
        """
        return self.results.describe()

    def load_results(self):
        """
        """
        pass

    def save_results(self): 
        pass
    
    def plot_results(self):
        pass

