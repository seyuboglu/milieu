"""GO Ontology data methods."""
import logging
import os
import json
import datetime
import time
import pickle
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from scipy.stats import spearmanr
import pandas as pd
from tqdm import tqdm
from goatools.obo_parser import GODag
from goatools.associations import read_ncbi_gene2go
from goatools.go_enrichment import GOEnrichmentStudy


def load_go_annotations(proteins,
                        levels=None,
                        obodag_path="data/go/go-basic.obo",
                        entrez_to_go_path="data/go/gene2go.txt"):
    """
    args:
    @proteins    (iterable)   proteins to get annotations for
    @levels  (list(int)) the levels of the ontology
    @obodag     (str)   path obo file
    @entrez_to_go_path (str) path to mapping from entrez ids to go doids

    return:
    @term_to_proteins (dict) map from term
    """
    obodag = GODag(obodag_path)
    entrez_to_go = read_ncbi_gene2go(entrez_to_go_path, taxids=[9606])

    def get_annotations(protein, levels):
        """
        """
        terms = set()
        doids = entrez_to_go[protein]
        for doid in doids:
            for parent in obodag[doid].get_all_parents():
                if levels is None or obodag[parent].level in levels:
                    terms.add(obodag[parent].name)

        return terms

    term_to_proteins = defaultdict(set)
    for protein in proteins:
        terms = get_annotations(protein, levels)
        for term in terms:
            term_to_proteins[term].add(protein)
    
    return term_to_proteins 