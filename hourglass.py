""" Functions for hourglass analysis based on paths. """
import operator
import functools
import collections

import networkx as nx


def path_centrality(paths, weighted=False):
    """ Calculate the path centrality for each node in a graph. 
    Args:
        paths (set/dict): set/dict of paths. set or dict.keys() contains
            the path (tuple of nodes), while dict.values() contains path 
            weights.
        weighted (bool): Whether to calculated weighted path centrality.
    Returns:
        pc (dict): Path centrality for each node in the connectome.
    """
    # paths must contain weights if specified weighted
    if weighted:
        assert isinstance(paths, dict)
        divide_path = lambda path: [(n, path[1]) for n in path[0]]
        node_centrality_list = functools.reduce(
            operator.iconcat, map(divide_path, paths.items()), []
        )
        pc = collections.defaultdict(lambda: 0)
        for key, value in node_centrality_list:
            pc[key] += value
        pc = dict(pc)
    else:
        assert isinstance(paths, (list, set))
        node_list = functools.reduce(operator.iconcat, list(paths), [])
        pc = dict(collections.Counter(node_list))

    return pc


def complexity(G, paths, weighted=False):
    """ Calculate the complexity for each node in a graph.
    Args:
        G (Network.DiGraph): Original graph where paths are generated
            from.
        paths (set): set of paths. Set contains
            the path (tuple of nodes).
        weighted (bool): Whether to caluclate weighted complexity.
    Returns:
        complexity (dict): Complexity for each node in the connectome.
    """
    to_v_paths = dict()
    for p in paths:
        for i, n in enumerate(p[1:]):
            if n not in to_v_paths:
                to_v_paths[n] = set()
            to_v_paths[n].add(tuple(p[: (i + 2)]))

    if weighted:
        complexity = dict()
        for n, ps in to_v_paths.items():
            complexity[n] = 0
            for p in ps:
                complexity[n] += functools.reduce(
                    lambda a, b: a * b,
                    [G[u][v]["weight"] for u, v in zip(p[:-1], p[1:])],
                )
        return complexity
    else:
        return {n: len(v) for n, v in to_v_paths.items()}

def generality(G, paths, weighted=False):
    """ Calculate the generality for each node in a graph.
    Args:
        G (Network.DiGraph): Original graph where paths are generated
            from.
        paths (set): set of paths. Set contains
            the path (tuple of nodes).
        weighted (bool): Whether to caluclate weighted generality.
    Returns:
        generality (dict): Generality for each node in the connectome.
    """
    from_v_paths = dict()
    for p in paths:
        for i, n in enumerate(p[:-1]):
            if n not in from_v_paths:
                from_v_paths[n] = set()
            from_v_paths[n].add(tuple(p[i:]))

    if weighted:
        generality = dict()
        for n, ps in from_v_paths.items():
            generality[n] = 0
            for p in ps:
                generality[n] += functools.reduce(
                    lambda a, b: a * b,
                    [G[u][v]["weight"] for u, v in zip(p[:-1], p[1:])],
                )
        return generality
    else:
        return {n: len(v) for n, v in from_v_paths.items()}


def find_tau_core(paths, tau=0.9, weighted=False):
    """ Find the smallest set of nodes (core nodes) which cover more than $\tau$
    of all paths. Weighted version finds the set of core nodes which cover 
    $\tau$ of all path weights. Use greedy algorithm which iteratively find
    nodes with highest path centrality.
    Args:
        paths (set/dict): set/dict of paths.
        tau (float): tau value.
        weighted (bool): whether to use weights.
    Returns:
        core_nodes (dict): core nodes ordered by contribution to coverage and 
            correponding incremental percentage of coverage.
    """
    # paths must contain weights if specified weighted
    if weighted:
        assert isinstance(paths, dict)
        total_path = sum(paths.values())
    else:
        total_path = len(paths)

    # Cache the Dictionary from node to the paths that they covered.
    nodes = {n for p in paths for n in p}
    node_path_dict = {n: set() for n in nodes}
    for p in paths:
        for n in p:
            node_path_dict[n].add(p)

    core_nodes = {}
    current_paths = paths
    current_path_centrality = collections.Counter(
        path_centrality(current_paths, weighted)
    )
    while sum(core_nodes.values()) < tau and len(current_paths) > 0:
        # Get the node with highest path centrality
        node = sorted(
            current_path_centrality.items(), key=lambda x: x[1], reverse=True
        )[0][0]
        # Record current coverage
        core_nodes[node] = current_path_centrality[node] / total_path
        # Get the paths to remove
        if weighted:
            paths_to_remove = {
                p: paths[p]
                for p in set(current_paths).intersection(node_path_dict[node])
            }
        else:
            paths_to_remove = set(current_paths).intersection(node_path_dict[node])
        # Reduce the path centrality calculated
        current_path_centrality = current_path_centrality - collections.Counter(
            path_centrality(paths_to_remove, weighted)
        )
        # Remove the paths that traverse the nodes in the tau-core
        if weighted:
            current_paths = {
                p: paths[p] for p in set(current_paths) - set(paths_to_remove)
            }
        else:
            current_paths = set(current_paths) - paths_to_remove

    return core_nodes


def create_flat_network(paths, weighted=False):
    """ Create the flat network from a set of paths and specified sources
    and targets. The flat network contains only sources and targets as nodes.
    Every ST-path from a source to a target is replaced by a direct connection. 
    Multiple connections between a set of source and target will be replaced
    with a single connection with the weight as sum of weights of original 
    connections.
    Args:
        paths (set/dict): set/dict of paths.
    Returns:
        flat_network (NetworkX.DiGraph): flat network constructed.
    """
    flat_network = nx.DiGraph()
    # Aggregate the path weight
    st_path_weights = {}
    for p in paths:
        s, t = p[0], p[-1]
        if (s, t) not in st_path_weights:
            st_path_weights[(s, t)] = 0
        if weighted:
            st_path_weights[(s, t)] += paths[p]
        else:
            st_path_weights[(s, t)] += 1

    # Create edge from st paths
    flat_network.add_weighted_edges_from(
        [(u, v, w) for (u, v), w in st_path_weights.items()]
    )

    return flat_network


def hourglass_score(paths, tau=0.9, weighted=False):
    """ Calculate the hourglass score. The hourglass score is defined as 
    $ H(\tau) = 1 - \frac{C(\tau)}{C_f{\tau}} $ where $C(\tau)$ is the size
    of tau core in original network, and $C_f(\tau)$ is the size of tau core
    in a flat network created.
    Args:
        paths (set/dict): set/dict of paths.
        tau (float): \tau used in tau-core calculation.
        weight (bool): whether to incorporate weights.
    Returns:
        hs (float): hourglass score.
    """
    core_size = len(find_tau_core(paths, tau, weighted))
    flat_network = create_flat_network(paths, weighted)
    # The created flat network should always have edge weight, therefore weights
    # are used in following analysis.
    flat_network_paths = {
        (u, v): data["weight"] for u, v, data in flat_network.edges(data=True)
    }
    flat_core_size = len(find_tau_core(flat_network_paths, weighted=True))

    return 1 - core_size / flat_core_size


def cumsum(vs):
    """ Return the cumulative sum of a list. 
    Args:
        vs (list): list of values.
    Returns:
        new_vs (list): list of cumulative sums.
    """
    new_vs = [vs[0]]
    for item in vs[1:]:
        new_vs.append(new_vs[-1] + item)
    return new_vs


def hourglass_scores(paths, max_tau=0.95, weighted=False):
    """ Calculate the hourglass scores with different values of tau until the
    max_tau is reached. The hourglass scores can be described as key turning
    points with flat line segments connecting between them. Return the key 
    turning points as the result.
    Args:
        paths (set/dict): set/dict for unweighted/weighted paths.
        max_tau (float): maximum of \tau to stop the calculation.
        weight (bool): whether to incorporate weight information or not.
    Returns:
        scores (list): hourglass score at each key turning point, in the form
            of list of (tau, score).
    """
    # Get cores
    cores = find_tau_core(paths, max_tau, weighted)
    # Get flat network cores
    flat_network = create_flat_network(paths, weighted)
    flat_network_paths = {
        (u, v): data["weight"] for u, v, data in flat_network.edges(data=True)
    }
    flat_network_cores = find_tau_core(flat_network_paths, max_tau, weighted=True)
    # Get cumulative core ratios
    cum_core_ratio = cumsum(list(cores.values())) + [1]
    cum_flat_core_ratio = cumsum(list(flat_network_cores.values())) + [1]
    # Get key points in the curve of hourglass score
    scores = [(0, 0)]
    core_index, flat_core_index = 0, 0
    while (core_index < len(cores)) or (flat_core_index < len(flat_network_cores)):
        ratio = min(cum_core_ratio[core_index], cum_flat_core_ratio[flat_core_index])
        scores.append((ratio, 1 - (core_index + 1) / (flat_core_index + 1)))
        if cum_core_ratio[core_index] > cum_flat_core_ratio[flat_core_index]:
            flat_core_index += 1
        elif cum_core_ratio[core_index] < cum_flat_core_ratio[flat_core_index]:
            core_index += 1
        else:
            flat_core_index += 1
            core_index += 1

    return scores


def area_under_curve(points, interpolate):
    """ Calculate the area under a curve based on the keypoints of the curve. 
    Args:
        points (list): list of (x, y).
        interpolate (str): methods used to interpolate between key points.
            Currently support `left`, `right`, `mean`.
    Returns:
        auc (float): area under curve
    """
    if interpolate == "left":
        return sum(
            [(x2 - x1) * y1 for (x1, y1), (x2, y2) in zip(points[:-1], points[1:])]
        )
    elif interpolate == "right":
        return sum(
            [(x2 - x1) * y2 for (x1, y1), (x2, y2) in zip(points[:-1], points[1:])]
        )
    elif interpolate == "mean":
        return sum(
            [
                (x2 - x1) * (y1 + y2) / 2
                for (x1, y1), (x2, y2) in zip(points[:-1], points[1:])
            ]
        )
    else:
        raise ValueError("Argument interpolate is not recognized.")

