""" Contains functions for randomization of connectomes. """
import random

import networkx as nx


def permute_edge_weight(connectome):
    """ Randomly permute weights of edges in the connectome, but retain the nodes,
    connections.
    Args:
        connectome (NetworkX.Digraph): original connectome.
    Returns:
        permuted_connectome (NetworkX.DiGraph): permuted connectome.
    """
    edge_weights = {
        (u, v): data["weight"] for u, v, data in connectome.edges(data=True)
    }
    # Shuffle the edge weights
    weights = list(edge_weights.values())
    random.shuffle(weights)
    # Assign back to the edges
    permuted_edge_weights = {e: v for e, v in zip(edge_weights.keys(), weights)}
    # Create a copy of the connectome with new weights
    permuted_connectome = connectome.copy()
    for e in connectome.edges():
        permuted_connectome.edges[e]["weight"] = permuted_edge_weights[e]

    return permuted_connectome


def randomize_sabrin(connectome, paths, weighted=False):
    """ Randomly reconnect edges, but retain the number of nodes, connections,
    in-degree of each node, and the partial ordering between nodes. For detailed
    explanation, check: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7029875/
    Args:
        connectome (NetworkX.Digraph): original connectome.
        paths (iterable of list): paths containing list of nodes as each entry.
        weighted (bool): Whether to keep edge weight during randomization.
    Returns:
        permuted_connectome (NetworkX.DiGraph): permuted connectome.
    """
    # Get ancestors
    prior_nodes = {n: set() for n in connectome.nodes()}
    after_nodes = {n: set() for n in connectome.nodes()}
    for p in paths:
        for i, n in enumerate(p):
            prior_nodes[n].update(p[:i])
            after_nodes[n].update(p[(i + 1) :])
    ancestors = {n: prior_nodes[n] - after_nodes[n] for n in connectome.nodes()}
    permuted_connectome = connectome.copy()
    for n in connectome.nodes():
        # if the node has no ancestors, no permutation could be performed.
        if len(ancestors[n]) == 0:
            continue

        in_edges = connectome.in_edges(n)
        new_us = random.choices(list(ancestors[n]), k=len(in_edges))
        if weighted:
            new_in_edges = [
                (new_u, n, connectome.edges[e]["weight"])
                for new_u, e in zip(new_us, in_edges)
            ]
            permuted_connectome.add_weighted_edges_from(new_in_edges)
        else:
            new_in_edges = [(new_u, n) for new_u in new_us]
            permuted_connectome.add_edges_from(new_in_edges)

        permuted_connectome.remove_edges_from(in_edges)

    return permuted_connectome
