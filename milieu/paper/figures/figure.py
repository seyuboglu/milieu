"""
Provides base class for all experiments 
"""
import os

import pandas 

from milieu.util.util import Process
 

class Figure(Process):
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

    def _run(self):
        """
        """
        self._generate()

    def _load_data(self):
        """
        """
        raise NotImplementedError
    
    def _generate(self):
        """
        """
        raise NotImplementedError

    def save(self): 
        """
        """
        raise NotImplementedError
    
    def show(self):
        """
        """
        raise NotImplementedError


