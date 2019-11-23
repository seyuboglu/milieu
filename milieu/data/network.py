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


class Network:
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
        node_names = set()
        edges = []
        with open(network_path) as network_file:
            for line in network_file:
                if remove_edges > 0 and random.random() < remove_edges:
                    continue
                p1, p2 = [int(a) for a in line.split()] 
                node_names.add(p1)
                node_names.add(p2)
                edges.append((p1, p2))
        if remove_nodes > 0: 
            assert(remove_nodes < 1)
            node_names = random.sample(node_names, 1 - remove_nodes)

        self.name_to_node = {p: n for n, p in enumerate(node_names)}
        self.node_to_name = {n: p for p, n in self.name_to_node.items()}

        # build adjacency matrix
        self.adj_matrix = np.zeros((len(self.name_to_node), 
                                    len(self.name_to_node)))
        for p1, p2 in edges:
            n1, n2 = self.name_to_node[p1], self.name_to_node[p2]
            self.adj_matrix[n1, n2] = 1
            self.adj_matrix[n2, n1] = 1
        
        self.nx = nx.from_numpy_matrix(self.adj_matrix)
    
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
        return protein in self.name_to_node
        
    def get_interactions(self, nodes=False):
        """
        Edges in are tuples in order 
        @param nodes (bool) returns edges as nodes if 
        """
        for a, b in self.nx.edges():
            if not nodes:
                a = self.node_to_name[a]
                b = self.node_to_name[b]
            
            yield (a, b) if a < b else (b, a)

    def get_name(self, node):
        """
        """
        return self.node_to_name.get(node, None)
    
    def get_names(self, nodes=None):
        """
        Converts an iterable of node ids to node names.
        args:
            nodes  (iterable)   List or other iterable of node ids
        return:
            proteins    (ndarray)
        """
        if nodes is not None:
            return np.array([self.node_to_name[n] 
                             for n in nodes if n in self.node_to_name])
        else:
            return np.fromiter(self.node_to_name.values(), dtype=int)
    
    def get_node(self, name):
        """
        """
        return self.name_to_node.get(name, None)
    
    def get_nodes(self, names=None):
        """
        Converts an iterable of protein ids to node ids. If protein is not in network, 
        it is omitted from the final np.array
        args:
            proteins  (iterable)   List or other iterable of protein ids
        return:
            proteins    (ndarray)
        """
        if names is not None:
            return np.array([self.name_to_node[p] 
                             for p in names if p in self.name_to_node])
        else:
            return np.fromiter(self.name_to_node.values(), dtype=int)
