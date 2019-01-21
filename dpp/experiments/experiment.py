"""
Provides base class for all experiments 
"""
import os
import pickle
import smtplib
import socket
import traceback

import pandas 

from dpp.util import parse_id_rank_pair, send_email
 

class Experiment(object):
    """ 
    Base class for all disease protein prediction methods.
    """
    def __init__(self, dir, params):
        """ Initialize the 
        Args: 
            dir (string) The directory where the experiment should be run
            params  (dict)
        """
        self.dir = dir 
        self.params = params
    
    def is_completed(self):
        """
        Checks if the experiment has already been run. 
        """
        return os.path.isfile(os.path.join(self.dir, 'results.csv'))

    def _run(self): 
        pass
    
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

    def load_results(self):
        """
        """
        pass

    def save_results(self): 
        pass
    
    def plot_results(self):
        pass


if __name__ == '__main__':
    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    json_path = os.path.join(args.dir, 'params.json')
    assert (os.path.isfile(json_path), 
            "No json configuration file found at {}".format(json_path))
    params = Params(json_path)
    params.update(json_path)  

    assert(hasattr(params, "source"))
    
    os.system(params.source + '--experiment_dir ' + args.dir)
