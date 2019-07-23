"""
Provides base class for all experiments 
"""
import os
import json

from milieu.process import Process
 

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

    
    def add_params(self, new_params):
        for key in new_params.keys():
            if key in self.params:
                raise ValueError("Cannot update existing parameter.")
            
        self.params.update(new_params)
        
        with open(os.path.join(self.dir, "params.json"), 'w') as f: 
            json.dump(self.params, f, indent=4)
    
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
