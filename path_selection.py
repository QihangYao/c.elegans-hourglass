""" Contains functions for path selection and path weighting. """
import re

import numpy as np
import networkx as nx
import scipy


def length_based_selection(
    connectome, sources, targets, tolerance=None, max_hops=None, verbose=False
):
    """ Select paths based on the length of the paths. The selection will
    start from identifying shortest path between sources and targets, 
    and select paths based on the its difference with shortest path and
    max number of edges.
    Args:
        connectome (NetworkX.DiGraph): graph to perform selection on.
        sources (iterable): source nodes.
        target (iterable): target nodes.
        tolerance (int): (optional) For each set of source and target, select paths
            with lengths <= tolerance + l_sp, where l_sp is the length of 
            shortest path between source and targets. If None, no limitation
            on the path length based on shortest path length.
        max_hops (int): (optional) if specified, only keep paths with length
            lower than max_hops. NOTE: It is strongly suggested that this
            limit is set, to control the algorithm running time.
        verbose (bool): Whether to print out the detailed path selection 
            process.
    Returns:
        paths (set): set of paths.
    """
    assert not ((tolerance is None) and (max_hops is None))
    shortest_paths = nx.shortest_path(connectome)
    shortest_st_paths = {
        (s, t): tuple(shortest_paths[s][t])
        for s in sources
        for t in targets
        if t in shortest_paths[s].keys()
    }
    paths = []
    current_paths = [(s,) for s in sources]
    while len(current_paths) > 0:
        p = current_paths.pop(0)
        for u, v in connectome.out_edges(p[-1]):
            new_p = p + (v,)
            # Don't allow repeating nodes in the path
            if v in p:
                continue
            # If reach any target
            if v in targets:
                # Check tolerance but not max hops (as the latter
                # is accomplished in further filtering)
                if (tolerance is not None) and (
                    len(new_p) > len(shortest_st_paths[p[0], v]) + tolerance
                ):
                    pass
                else:
                    paths.append(new_p)
                    if verbose == True:
                        print(len(paths))
            # Check max hops limit if exist
            if max_hops:
                if len(new_p) >= max_hops + 1:
                    continue
            # Check tolerance limit (for any t, the current path length
            # + distance to t <= shortest s-t length + tolerance)
            if tolerance is not None:
                if not any(
                    [
                        (len(shortest_st_paths[(p[0], t)]) + tolerance)
                        >= (len(shortest_paths[v][t]) + len(p))
                        for t in targets
                        if t in shortest_paths[v].keys()
                    ]
                ):
                    continue
            current_paths.append(new_p)

    return set(paths)


def parse_select(connectome, sources, targets, method_str):
    """ Parse the string for path selection method and select paths based on
    the method.
    Args:
        connectome (NetworkX.DiGraph): graph to perform selection on.
        sources (iterable): source nodes.
        target (iterable): target nodes.
        method_str (str or list of strs): string used to indicate the method
            to be used. Currently support "length,sp+ApB" to apply 
            length-based selection. This would result in including only 
            paths A hops longer than shorted path between the same source
            and target, and not longer than B hops.
    Returns:
        paths (list of tuples): the list of paths selected.
    """
    assert isinstance(method_str, (str, list))
    if isinstance(method_str, str):
        method_str = [method_str]
    candidate_paths = []
    for entry in method_str:
        method, args = entry.split(",")
        if method == "length":
            tolerance, max_hops = None, None
            tol_regex = re.compile("sp\+([1-9][0-9]*|0)")
            result = tol_regex.search(args)
            if result is not None:
                tolerance = int(result.group(1))
            print(tolerance)
            max_hops_regex = re.compile("p([1-9][0-9]*|0)")
            result = max_hops_regex.search(args)
            if result is not None:
                max_hops = int(result.group(1))
            candidate_paths.append(
                length_based_selection(
                    connectome, sources, targets, tolerance, max_hops
                )
            )
        else:
            raise NotImplementedError

    return set.intersection(*candidate_paths)


def met(
    connectome,
    paths,
    edge_normalize=None,
    edge_transform=None,
    path_normalize=None,
    path_transform=None,
):
    """ Multi-edge transformation. The idea is to transform the each edge 
    so that an edge with weight $w$ will become $w$ edges with weight $1$.
    Args:
        connectome (NetworkX.DiGraph): graph to perform selection on.
        paths (list): list of paths.
        edge_normalize (func): (optional) If specified, the weight of each edge
            will be normalized based on the function provided. The function
            accept a set of weights and return a value for normalization.
        edge_transform (func): (optional) If specified, transform the edge 
            weight using the function. The function accept a single weight
            value and return a transformed value. 
        path_normalize (func): (optional) If specified, the weight of path
            will be normalized based on the function provided. The function
            accept weight of edges along a path as input and return a value
            for normalization.
        path_transform (func): (optional) If specified, transform the path
            weight using the function. The function accept a single weight
            value and return a transformed value.
    Returns:
        path_weights (dict): dictionary with path as key and weight as value.
    """
    # Avoid using two additional methods together or we will get negative
    # values.
    edge_weights = {
        (u, v): data["weight"] for u, v, data in connectome.edges(data=True)
    }
    if edge_normalize is not None:
        edge_normalize_value = edge_normalize(list(edge_weights.values()))
    path_weights = {}
    for p in paths:
        weights = np.array([edge_weights[(u, v)] for u, v in zip(p[:-1], p[1:])])
        if edge_normalize is not None:
            weights = weights / edge_normalize_value
        if edge_transform is not None:
            weights = edge_transform(weights)
        path_weight = np.prod(weights)
        if path_transform is not None:
            path_weight = path_transform(path_weight)
        if path_normalize is not None:
            path_normalize_value = path_normalize(weights)
            path_weight = path_weight / path_normalize_value
        path_weights[p] = path_weight

    return path_weights


def geometric_mean(connectome, paths):
    """ Calculate the weight for each path through taking geometric mean of
    weight of all edges.
    Args:
        connectome (NetworkX.DiGraph): graph to perform selection on.
        paths (iterable): paths to include.
    Returns:
        path_weights (dict): dictionary with path as key and weight as value.
    """
    edge_weights = {
        (u, v): data["weight"] for u, v, data in connectome.edges(data=True)
    }
    path_weights = {}
    for p in paths:
        weights = [edge_weights[(u, v)] for u, v in zip(p[:-1], p[1:])]
        # Use Log transformation to avoid possible overflow.
        # (Given the range of edge weights.)
        path_weights[p] = np.exp(np.log(weights).sum() / len(weights))

    return path_weights


def flow(connectome, paths):
    """ Calculate the flow of each path as path weights. The flow is 
    calculated as the minimal weight of edges along a path.
    Args:
        connectome (NetworkX.DiGraph): graph to perform selection on.
        paths (iterable): paths to include.
    Returns:
        path_weights (dict): dictionary with path as key and weight as value.
    """
    edge_weights = {
        (u, v): data["weight"] for u, v, data in connectome.edges(data=True)
    }
    path_weights = {}
    for p in paths:
        weights = [edge_weights[(u, v)] for u, v in zip(p[:-1], p[1:])]
        path_weights[p] = np.min(weights)

    return path_weights


def fidelity(connectome, paths, weighted=False):
    """ Calculate the information fidelity of each path as path weights.
    The information fidelity is defined as the product of information
    preserved in information flow from source to target in each edge in a
    path. The ratio information preserved by information flow through an 
    edge is calculated as the edge weight divided by (weighted) in-degree
    of the target.
    Args:
        connectome (NetworkX.DiGraph): graph to perform selection on.
        paths (iterable): paths to include.
        weighted (bool): whether to incorporate weight information.
    Returns:
        path_weights (dict): dictionary with path as key and weight as value.
    """
    path_weights = {}
    for p in paths:
        if weighted:
            preserved_ratio = [
                connectome.edges[(u, v)]["weight"]
                / connectome.in_degree(v, weight="weight")
                for u, v in zip(p[:-1], p[1:])
            ]
        else:
            preserved_ratio = [
                1 / connectome.in_degree(v) for u, v in zip(p[:-1], p[1:])
            ]
        path_weights[p] = np.prod(preserved_ratio)

    return path_weights


def frequency(connectome, paths):
    """ Assuming that edge weight is a relative representation of the 
    frequency of that edge being used, and this frequency is equal to sum
    of frequencies of all paths that include that edge, we can calculate the
    weight of paths using the weight of edge and their correspondence matrix.
    Args:
        connectome (NetworkX.DiGraph): graph to perform selection on.
        paths (iterable): paths to include.
    Returns:
        path_weights (dict): dictionary with path as key and weight as value.
    """
    edge_weights = {
        (u, v): data["weight"] for u, v, data in connectome.edges(data=True)
    }
    # Normalize the edge weights and take the values
    edge_weights = np.array(
        [edge_weights[e] / sum(edge_weights.values()) for e in edge_weights]
    )
    edge_indices = {e: i for i, e in enumerate(connectome.edges())}
    # Create edge-path correspondance matrix
    row_inds, col_inds = [], []
    for i, p in enumerate(paths):
        for e in zip(p[:-1], p[1:]):
            row_inds.append(edge_indices[e])
            col_inds.append(i)

    edge_path_corr = scipy.sparse.csr_matrix(
        (np.ones(len(row_inds)), (row_inds, col_inds))
    )

    result = scipy.optimize.lsq_linear(
        edge_path_corr, edge_weights, bounds=(0, np.inf), tol=1e-5
    )
    return {p: x for p, x in zip(paths, result.x)}


def uniform(paths, constant=1):
    """ Assign uniform weight to the paths. """
    return {p: constant for p in paths}


def parse_weighting(connectome, paths, weighting_str):
    """ Weight the paths based on the path weighting method specified. 
    Args:
        paths (iterable of tuples): paths selected.
        weighting_str (str): string that specified the path weighting method,
            currently support "met", "met-median", "met-log", "gm", "flow",
            "fidelity", "frequency".
    Returns:
        path_weights (dict): dictionary with path as key and weight as value.
    """
    if weighting_str == "met":
        return met(connectome, paths)
    elif weighting_str == "met-median":
        return met(connectome, paths, edge_normalize=np.median)
    elif weighting_str == "met-log":
        return met(connectome, paths, edge_transform=np.log)
    elif weighting_str == "met-len-norm":
        return met(connectome, paths, path_normalize=len, path_transform=np.log)
    elif weighting_str == "met-len-norm-median":
        return met(
            connectome,
            paths,
            edge_normalize=np.median,
            path_normalize=len,
            path_transform=np.log,
        )
    elif weighting_str == "gm":
        return geometric_mean(connectome, paths)
    elif weighting_str == "flow":
        return flow(connectome, paths)
    elif weighting_str == "fidelity":
        return fidelity(connectome, paths, weighted=True)
    elif weighting_str == "frequency":
        return frequency(connectome, paths)
    elif weighting_str == "uniform":
        return uniform(paths)
    else:
        raise NotImplementedError

