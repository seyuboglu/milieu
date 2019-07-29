
import ndex2.client as nc
from cyjupyter import Cytoscape
import networkx as nx

from milieu.util import load_mapping


def show_network(network, seed_proteins, pred_proteins, 
                 id_format="genbank", style=None, show_seed_mi=True):
    """
    """
    if id_format == "genbank":
        genbank_to_entrez = load_mapping("data/protein/genbank_to_entrez.txt",
                                         b_transform=int, delimiter='\t')
        seed_proteins = [genbank_to_entrez[protein] 
                         for protein in seed_proteins if protein in genbank_to_entrez]
        pred_proteins = [genbank_to_entrez[protein] 
                         for protein in pred_proteins if protein in genbank_to_entrez]
        seed_nodes = network.get_nodes(seed_proteins)
        pred_nodes = network.get_nodes(pred_proteins) 
    
    elif id_format == "entrez":
        seed_nodes = network.get_nodes(seed_proteins)
        pred_nodes = network.get_nodes(pred_proteins) 
    
    else:
        raise ValueError("id_format is not recognized.")

    cyjs_network = get_network(network, seed_nodes, pred_nodes, show_seed_mi)
    # Unique ID for a network entry in NDEx
    uuid ='f28356ce-362d-11e5-8ac5-06603eb7f303'

    # NDEx public server URL
    ndex_url = 'http://public.ndexbio.org/'

    # Create an instance of NDEx client
    ndex = nc.Ndex2(ndex_url)

    # Download the network in CX format
    response = ndex.get_network_as_cx_stream(uuid)

    # Store the data in a Python object
    cx = response.json()

    if style is None:
        style = [
            {
                "selector": "node",
                "css": {
                    "content": "data(genbank)",
                    "border-color" : "rgb(256,256,256)",
                    "border-opacity" : 1.0,
                    "border-width" : 2,

                },
            },
            {
                "selector": "node[role = 'seed']",
                "css": {
                    "background-color": "#f53e37",
                    "width": 20, 
                    "height": 20
                }, 
            },
            {
                "selector": "node[role = 'pred']",
                "css": {
                    "background-color": "#ff9529",
                    "width": 20, 
                    "height": 20
                }, 
            },
            {
                "selector": "node[role = 'mutual_interactor']",
                "css": {
                    "background-color": "#6599d1",
                    "width": 20, 
                    "height": 20
                }, 
            }
        ]

    return Cytoscape(data=cyjs_network, visual_style=style)


def get_network(network, seed_nodes, pred_nodes, show_seed_mi=True):
    """ Get the disease subgraph of 
    Args:
        disease: (Disease) A disease object
    """
    entrez_to_genbank = load_mapping("data/protein/genbank_to_entrez.txt",
                                     b_transform=int, delimiter='\t', reverse=True)
                    
    nodes = {}

    def add_node(node, role="seed"):
        if node not in nodes:
            nodes[node] = { 
                "data": {
                    "role": role, 
                    "id": node,
                    "entrez": network.get_protein(node),
                    "genbank": entrez_to_genbank.get(network.get_protein(node), "")
                }
            }

    # add seed nodes
    for seed_node in seed_nodes:
        add_node(seed_node, role="seed")

    # get seed node neighbors
    seed_node_to_nbrs = {node: set(network.nx.neighbors(node)) 
                         for node in seed_nodes}
    # get mutual interactors between preds and seed
    for pred_node in pred_nodes:
        add_node(pred_node, role="pred")
    for pred_node in pred_nodes:
        pred_nbrs = set(network.nx.neighbors(pred_node)) 
        for seed_node in seed_nodes:
            seed_nbrs = seed_node_to_nbrs[seed_node]
            common_nbrs = seed_nbrs & pred_nbrs
            for common_nbr in common_nbrs:
                add_node(common_nbr, role="mutual_interactor")

    # the set of nodes intermediate between nodes in the 
    if show_seed_mi:
        for a, node_a in enumerate(seed_nodes):
            for b, node_b in enumerate(seed_nodes):
                # avoid repeat pairs
                if a >= b:
                    continue
                common_nbrs = seed_node_to_nbrs[node_a] & seed_node_to_nbrs[node_b]
                for common_nbr in common_nbrs:
                    add_node(common_nbr, role="mutual_interactor")
    
    # get induced subgraph 
    subgraph = nx.Graph(network.nx.subgraph(nodes.keys()))

    subgraph.remove_edges_from(subgraph.selfloop_edges())

    edges = []
    for edge in subgraph.edges():
        edges.append({
            "data": {
                "source": edge[0],
                "target": edge[1]
            }
        })

    return {"elements": {"nodes": list(nodes.values()), "edges": edges}}