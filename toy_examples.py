""" Provide toy example classes which we can test PathSelection on. """

import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt


class ToyExample(object):
    """ Class as template for all DAG toy examples. These toy examples could
    be used to test whether a path weighting model achieves the expected
    result. """

    def __init__(self):
        self.G = None  # Create a graph.
        self.sources = None
        self.targets = None
        raise Exception("Instance of toyexample unexpected.")

    def display(self, path=None):
        """ Display the toy example. 
        Args:
            path (str): (optional) Result image file path.
        """
        node_colors = {n: "b" for n in self.G.nodes()}
        for s in self.sources:
            node_colors[s] = "g"
        for t in self.targets:
            node_colors[t] = "r"

        pos = graphviz_layout(self.G, prog="dot", args="-Grankdir=BT")
        # pos = {n: (y, x) for n, (x, y) in pos.items()}
        fig, ax = plt.subplots()
        nx.draw(
            self.G,
            with_labels=True,
            pos=pos,
            node_color=list(node_colors.values()),
            font_color="white",
            edge_color="grey",
            # width=[data["weight"] for _, _, data in self.G.edges(data=True)],
        )
        nx.draw_networkx_edge_labels(
            self.G,
            pos=pos,
            edge_labels={
                (u, v): data["weight"] for u, v, data in self.G.edges(data=True)
            },
            ax=ax,
            rotate=False
        )
        if path:
            plt.savefig(path, dpi=300)

    def get_graph(self):
        """ Get the connectome, sources and targets.
        Returns:
            G (NetworkX.DiGraph): toy example graph.
            sources (iterable): source nodes.
            target (iterable): target nodes.
        """
        return self.G, self.sources, self.targets

    def test(self, path_centrality):
        """ Run through some tests. Accept the path weights as the input. 
        Args:
            path_centrality (dict): node as key and path centrality as value.
        Returns:
            passed (bool): Whether all tests are passed.
            error (str or NoneType): Error message if not passed.
        """
        raise Exception("Test not implemented for object %s" % type(self))


class BackBone(ToyExample):
    """ 3-layer fully connected network, with 4 source nodes, 
    4 targets, and 2 intermediate nodes. The edge weight are consistently 1
    among all edges.
    """

    def __init__(self):
        self.sources = range(4)
        self.targets = range(6, 10)
        self.G = build_fully_connected_network([4, 2, 4])

    def test(self, path_centrality):
        """ A series of tests on backbone network. """
        # Nodes should share same centrality if they are in the same layer.
        if not all(
            [
                all(
                    [
                        path_centrality[n1] == path_centrality[n2]
                        for n1, n2 in zip(layer[:-1], layer[1:])
                    ]
                )
                for layer in [range(4), range(4, 6), range(6, 10)]
            ]
        ):
            return (
                False,
                "BackBone Test: Nodes in same layer don't have same path centrality.",
            )
        # Source and targets should have same path centrality.
        if not path_centrality[0] == path_centrality[-1]:
            return (
                False,
                "BackBone Test: Sources and targets don't have same path centrality.",
            )
        # Intermediate nodes should have higher weights
        if not path_centrality[4] > path_centrality[0]:
            return (
                False,
                "BackBone Test: Intermediate nodes don't have higher path centrality.",
            )
        return True, None


class SmallBranch(ToyExample):
    """ Add a non-path forming branch to an intermediate node in the backbone
    shouln't or shouldn't significantly change the path centrality of that node.
    """

    def __init__(self):
        self.sources = range(4)
        self.targets = range(6, 10)
        self.G = add_small_branch(BackBone().G, 5)

    def test(self, path_centrality, tolerance=0.9):
        """ Test: The path centrality of intermediate nodes with that branch
        and without that branch should be equal or similar.
        Args:
            path_centrality (dict): node as key and path centrality as value.
        """
        # Test if path centrality of two intermediate nodes are equal or the
        # intermediate node with branch have lower path centrality but within
        # tolerance.
        if (
            tolerance * path_centrality[4] <= path_centrality[5]
            and path_centrality[4] >= path_centrality[5]
        ):
            return True, None
        else:
            return (
                False,
                "SmallBranch Test: Small branch shouldn't change the path centrality a lot.",
            )


class IshaanExample(ToyExample):
    """ Toy example used in Ishaan's old draft. """

    def __init__(self):
        self.sources = ["a"]
        self.targets = ["p", "q", "r"]
        self.G = nx.DiGraph()
        edges = [
            ("a", "b", 2),
            ("a", "c", 2),
            ("a", "d", 4),
            ("a", "e", 4),
            ("b", "f", 1),
            ("c", "f", 1),
            ("d", "g", 1),
            ("e", "g", 1),
            ("f", "h", 5),
            ("f", "i", 1),
            ("f", "j", 1),
            ("f", "k", 1),
            ("g", "l", 1),
            ("g", "m", 4),
            ("h", "p", 2),
            ("i", "n", 1),
            ("j", "n", 1),
            ("k", "n", 1),
            ("n", "p", 1),
            ("n", "q", 1),
            ("l", "o", 1),
            ("o", "q", 1),
            ("o", "r", 1),
            ("m", "r", 2),
        ]
        self.G.add_weighted_edges_from(edges)

class IshaanExampleModified(ToyExample):
    """ Toy example used in Ishaan's old draft. """

    def __init__(self):
        self.sources = ["a"]
        self.targets = ["l", "m", "n"]
        self.G = nx.DiGraph()
        edges = [
            ("a", "b", 2),
            ("a", "c", 4),
            ("b", "d", 5),
            ("b", "e", 1),
            ("b", "f", 1),
            ("b", "g", 1),
            ("c", "h", 1),
            ("c", "i", 4),
            ("d", "l", 2),
            ("e", "j", 1),
            ("f", "j", 1),
            ("g", "j", 1),
            ("j", "l", 1),
            ("j", "m", 1),
            ("h", "k", 1),
            ("k", "m", 1),
            ("k", "n", 1),
            ("i", "n", 2),
        ]
        self.G.add_weighted_edges_from(edges)

class QihangExample(ToyExample):
    """ Toy example used in Ishaan's old draft. """

    def __init__(self):
        self.sources = ["a", "b"]
        self.targets = ["h", "i", "j"]
        self.G = nx.DiGraph()
        edges = [
            ("a", "c", 2),
            ("a", "d", 1),
            ("b", "d", 1),
            ("b", "e", 2),
            ("c", "f", 2),
            ("d", "f", 1),
            ("d", "g", 1),
            ("e", "g", 2),
            ("f", "h", 1),
            ("f", "i", 1),
            ("g", "i", 1),
            ("g", "j", 1),
        ]
        self.G.add_weighted_edges_from(edges)

def build_fully_connected_network(layers):
    """ Build fully connected network with constant weight 1. 
    Args:
        layers (list): number of neurons in each layer.
    Returns:
        G (NetworkX.DiGraph): The network built.
    """
    G = nx.DiGraph()
    layer_nodes = [
        [sum(layers[:i]) + n for n in range(layer)] for i, layer in enumerate(layers)
    ]
    # Add all nodes
    G.add_nodes_from([n for layer in layer_nodes for n in layer])
    # Add all edges
    edges = [
        [s, t, 1]
        for sources, targets in zip(layer_nodes[:-1], layer_nodes[1:])
        for s in sources
        for t in targets
    ]
    G.add_weighted_edges_from(edges)

    return G


def add_small_branch(G, branch_node):
    """ Example1: What if we add a small branch at a intermediate node. """
    G = G.copy()
    new_node = len(G.nodes)
    G.add_node(new_node)
    G.add_edge(branch_node, new_node, weight=0.1)
    return G


def double_node_connectivity(G, target_node):
    """ Example2: double the weights of edges connected to a node. """
    G = G.copy()
    for edge in G.in_edges(target_node):
        G.edges[edge]["weight"] = G.edges[edge]["weight"] * 3
    for edge in G.out_edges(target_node):
        G.edges[edge]["weight"] = G.edges[edge]["weight"] * 3
    return G


def replace_node_with_edge(G, split_node, weight_agg_func):
    """ Example3: split a node into two connected nodes. """
    G = G.copy()
    new_node = len(G.nodes)
    G.add_node(new_node)
    out_edges = list(G.out_edges(split_node))
    G.add_edge(
        split_node,
        new_node,
        weight=weight_agg_func([G.edges[e]["weight"] for e in out_edges]),
    )
    for u, v in out_edges:
        G.add_edge(new_node, v, weight=G.edges[u, v]["weight"])
        G.remove_edge(u, v)

    return G


def display_network(G, sources, targest, path=None):
    node_colors = np.array(["b" for _ in G.nodes()])
    node_colors[sources] = "r"
    node_colors[targets] = "g"

    pos = graphviz_layout(G, prog="dot", args="-Grankdir=LR")
    # pos = nx.spring_layout(G)
    fig, ax = plt.subplots()
    nx.draw(
        G,
        with_labels=True,
        pos=pos,
        node_color=node_colors,
        font_color="white",
        width=[data["weight"] for _, _, data in G.edges(data=True)],
    )
    if path:
        plt.savefig(path, dpi=300)

