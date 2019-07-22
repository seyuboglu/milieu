"""
Module for loading and storing protein-protein interaction networks. 
"""

import os
import logging
import random

import numpy as np 
import pandas as pd 
import networkx as nx
#from goatools.obo_parser import GODag


class PPINetwork:
    """
    Represents a protein protein interaction network. 
    """

    def __init__(self, network_path, remove_edges=0, remove_nodes=0):
        """
        Load a protein-proetin interaction network from an adjacency list.
        args:
            network_path (string)
            remove_edges (double) fraction between 0 and 1 inclusive indicating 
            fraction of edges to randomly remove 
            remove_nodes (double) fraction between 0 and 1 inclusive indicating 
            fraction of nodes to randomly remove 
        """
        # map protein entrez ids to node index
        protein_ids = set()
        with open(network_path) as network_file:
            for line in network_file:
                p1, p2 = [int(a) for a in line.split()]
                protein_ids.add(p1)
                protein_ids.add(p2)

        self.protein_to_node = {p: n for n, p in enumerate(protein_ids)}
        self.node_to_protein = {n: p for p, n in self.protein_to_node.items()}

        # build adjacency matrix
        self.adj_matrix = np.zeros((len(self.protein_to_node), 
                                    len(self.protein_to_node)))
        with open(network_path) as network_file:
            for line in network_file:
                p1, p2 = [int(a) for a in line.split()]
                n1, n2 = self.protein_to_node[p1], self.protein_to_node[p2]
                self.adj_matrix[n1, n2] = 1
                self.adj_matrix[n2, n1] = 1
        
        self.nx = nx.from_numpy_matrix(self.adj_matrix)

        # remove random nodes, for robustness tests
        if remove_nodes > 0:
            assert(remove_nodes < 1)
            for _ in range(int(len(self.nx) * remove_nodes)):
                node = random.choice(list(self.nx.nodes()))
                self.nx.remove_node(node)
                protein = self.node_to_protein[node]
                del self.protein_to_node[protein]
                del self.node_to_protein[node]

        if remove_edges > 0:
            assert(remove_edges < 1)
            edges = random.sample(self.nx.edges(), k=int(self.nx.number_of_edges() * 
                                                         remove_edges))
            self.nx.remove_edges_from(edges)
            # can't have isolates in the graph
            isolates = nx.isolates(self.nx)
            for isolate in isolates:
                self.nx.remove_node(isolate)
                protein = self.node_to_protein[isolate]
                del self.protein_to_node[protein]
                del self.node_to_protein[isolate]

        self.adj_matrix = nx.to_numpy_matrix(self.nx)
    
    def __len__(self):
        """
        Returns the size of the network.
        """
        return len(self.nx)
    
    def __contains__(self, protein):
        """
        True if protein is in network. 
        @protein (int) entrez id for a protein
        """
        return protein in self.protein_to_node
        
    def get_interactions(self, nodes=False):
        """
        Edges in are tuples in order 
        @param nodes (bool) returns edges as nodes if 
        """
        for a, b in self.nx.edges():
            if not nodes:
                a = self.node_to_protein[a]
                b = self.node_to_protein[b]
            
            yield (a, b) if a < b else (b, a)

    def get_protein(self, node):
        """
        """
        return self.node_to_protein.get(node, None)
    
    def get_proteins(self, nodes=None):
        """
        Converts an iterable of node ids to protein ids.
        args:
            nodes  (iterable)   List or other iterable of node ids
        return:
            proteins    (ndarray)
        """
        if nodes is not None:
            return np.array([self.node_to_protein[n] 
                             for n in nodes if n in self.node_to_protein])
        else:
            return np.fromiter(self.node_to_protein.values(), dtype=int)
    
    def get_node(self, protein):
        """
        """
        return self.protein_to_node.get(protein, None)
    
    def get_nodes(self, proteins=None):
        """
        Converts an iterable of protein ids to node ids. If protein is not in network, 
        it is omitted from the final np.array
        args:
            proteins  (iterable)   List or other iterable of protein ids
        return:
            proteins    (ndarray)
        """
        if proteins is not None:
            return np.array([self.protein_to_node[p] 
                             for p in proteins if p in self.protein_to_node])
        else:
            return np.fromiter(self.protein_to_node.values(), dtype=int)
