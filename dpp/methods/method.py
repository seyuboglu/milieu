""" Defines DPPMethod
"""

class DPPMethod(object):
    """ 
    Base class for all disease protein prediction methods.
    """
    def __init__(self, ppi_network, diseases, params):
        """
        """
        self.network = ppi_network
        self.diseases = diseases
        self.params = params

    def compute_scores(self, train_nodes, disease): 
        pass

    def __call__(self, train_nodes, disease): 
        return self.compute_scores(train_nodes, disease)