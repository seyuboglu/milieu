"""
Utility functions for loading drug target data.  
"""
import os
from collections import defaultdict

import numpy as np 
import pandas as pd 


def load_drug_targets(path, delimiter="\t"):
    """
    args:
    @path   (str)   path to file with each line formatted [drug][delimiter][target]
    @delimiter  (str)   string separating drug and target in file

    return:
    @drug_to_targets    (dict)  maps drug to a set of target entrez ids 
    """
    drug_to_targets = defaultdict(set)
    with open(path) as f:
        for line in f:
            if line[0] == '#':
                continue
            drug, target = line.split(delimiter)
            target = int(target.strip("\n"))
            
            if not drug or not target:
                continue

            drug_to_targets[drug].add(target)

    return drug_to_targets 
