""" Main script and functions for hourglass analysis on weighted C.elegans
connectome. Each function in this module indicates a standalone analysis and
should be able to run independently, possibly with several simple arguments.
However, functions starting with `_` are functions built for speedup or code
reduction, and shouldn't be called independently. """

import bisect
import os
import functools
import multiprocessing

import numpy as np
import pandas as pd
import scipy

import data
import path_selection
import summary
import hourglass
import randomize


MAX_TAU = 0.95


def compare_count_size_connectome(sex, connection_type, preprocess=False):
    """ Compare the count and the size of the connectome. 
    Args:
        sex (str): sex of the connectome to be compared. Currently support ["herm", "male"].
        connection_type (str): type of the connection, which could be either
            "chem synapse" or "gap jn"
        preprocess (bool): whether to preprocess using data.preprocess before
            comparison
    """
    connectomes = {
        w_type: data.get_connectome(sex, [connection_type], [w_type], "graph")
        for w_type in data.WEIGHT_TYPES
    }
    if preprocess:
        connectomes = {
            w_type: data.preprocess(c, weight=" ".join((connection_type, w_type)))
            for w_type, c in connectomes.items()
        }
    # Compare the edges available
    for origin_type, new_type in zip(data.WEIGHT_TYPES, data.WEIGHT_TYPES[::-1]):
        print(
            "Comparing to %s connectome, the %s connectome have:"
            % (origin_type, new_type)
        )
        edges = set(connectomes[new_type].edges()) - set(
            connectomes[origin_type].edges()
        )
        nodes = set(connectomes[new_type].nodes()) - set(
            connectomes[origin_type].nodes()
        )
        print("%d unique edges." % (len(edges)))
        print("%d unique nodes, with type distribution" % (len(nodes)))
        node_types = [
            connectomes[new_type].nodes[n]["cell type"]
            for n in nodes
            if "cell type" in connectomes[new_type].nodes[n]
        ]
        if len(node_types) > 0:
            print(
                {
                    v: count
                    for v, count in zip(*np.unique(node_types, return_counts=True))
                }
            )
        else:
            print("Not Clear")

    # Compare the weight distribution
    for w_type in data.WEIGHT_TYPES:
        print("For weight type %s" % w_type)
        attr_name = " ".join((connection_type, w_type))
        weights = [
            attrs[attr_name] for u, v, attrs in connectomes[w_type].edges(data=True)
        ]
        print(scipy.stats.describe(weights))


def _calculate_hourglass(connectome, paths, weighting_str):
    """ Calculate hourglass AUC on a specified connectome with specified set
    of paths and specified path weighting methods. Designed as internal function
    for repeated calling during randomization test.
    Args:
        connectome (NetworkX.DiGraph): graph to perform selection on.
        paths (list): list of paths.
        weighting_str (str): string that specified the path weighting method,
            currently support "met", "met-median", "met-log", "gm", "flow",
            "fidelity", "frequency".
    Returns:
        hourglass_scores (list): hourglass score at each key turning point, in 
            the form of list of (tau, score).
        hourglass_auc (float): area under h-score curve.
    """
    path_weights = path_selection.parse_weighting(connectome, paths, weighting_str)
    hourglass_scores = hourglass.hourglass_scores(
        path_weights, max_tau=MAX_TAU, weighted=True
    )
    hourglass_auc = hourglass.area_under_curve(hourglass_scores, interpolate="left")
    return hourglass_scores, hourglass_auc


def _map_wrapper(func, data, multi_args=False, multi_process=False):
    """ Wrapper for multi-processing. Apply the func on each entry of data. 
    Args:
        func (function): function to map onto data.
        data (list): data as list of entries
        multi_args (bool): whether there are multiple arguments present.
        multi_process (bool): whether to apply multiprocessing
    """
    if multi_process:
        pool = multiprocessing.Pool()
        if multi_args:
            return pool.starmap(func, data)
        else:
            return pool.map(func, data)
    else:
        if multi_args:
            return [func(*entry) for entry in data]
        else:
            return [func(entry) for entry in data]


def compare_path_weighting(
    sex,
    connection_type,
    weight_type,
    path_select_method,
    path_weighting_methods,
    num_random_samples=100,
    multi_process=False,
):
    """ Compare the path weighting methods on a fixed connectome.
    Args:
        sex (str): sex of target connectome. Currently support ["herm", "male"].
        connection_type (str): target connection type. Currently support 
            ["chem synapse", "gap junction"].
        weight_type (str): target weight type. Currently support ["count", "size"].
        path_select_method (str): string that specified how path 
            selection should be conducted. Currently support "length:sp+ApB" 
            to apply length-based selection. This would result in including
            only paths A hops longer than shorted path between the same source
            and target, and not longer than B hops. 
        path_weighting_methods (list): list of strings that specified the path
            weighting method, currently support "met", "met-median", "met-log",
            "gm", "flow", "fidelity", "frequency".
        num_random_samples (int): number of randomized networks generated to 
            compare the result on empirical network with.
        multi_process (bool): Whether to apply multi-processing during 
            randomization.
    """
    print(
        "Sex: %s, Connection Type: %s, Weight Type: %s, Path Selection Method: %s"
        % (sex, connection_type, weight_type, path_select_method)
    )
    figure_prefix = os.path.join(
        "figs",
        "%s-%s-%s-" % tuple(map(data.get_abbr, (sex, connection_type, weight_type))),
    )
    # Get the connectome
    connectome = data.get_connectome(sex, [connection_type], [weight_type])
    # Preprocess
    connectome = data.preprocess(connectome)
    connectome = data.remove_feedback_edges(connectome)
    # Get paths based on the path selection
    S, I, M = data.get_sim_neurons(connectome)
    paths = path_selection.parse_select(connectome, S, M, path_select_method)
    # Generate randomized networks from the connectome for robustness analysis
    randomized_connectomes = [
        randomize.permute_edge_weight(connectome) for _ in range(num_random_samples)
    ]
    agg_core_nodes = dict()
    agg_hourglass_scores = dict()
    # Get path weighting result with each method
    for i, weighting_method in enumerate(path_weighting_methods):
        print("[%d] Weighting method: %s" % (i + 1, weighting_method))
        path_weights = path_selection.parse_weighting(
            connectome, paths, weighting_method
        )
        # Show the summary of path weights
        summary.path_weight_statistics(path_weights)
        # Core nodes / Hourglass scores at \tau=0.9
        core_nodes = hourglass.find_tau_core(path_weights, tau=0.9, weighted=True)
        agg_core_nodes[weighting_method] = core_nodes
        print("--- Core nodes at tau=0.9 ---")
        summary.core_nodes_info(connectome, core_nodes)
        summary.core_nodes_path_info(
            core_nodes,
            path_weights,
            path=figure_prefix
            + "%s-core-paths-weight-dist.png" % data.get_abbr(weighting_method),
        )
        # Hourglass scores and area under curve
        h_scores = hourglass.hourglass_scores(
            path_weights, max_tau=MAX_TAU, weighted=True
        )
        agg_hourglass_scores[weighting_method] = h_scores
        hourglass_auc = hourglass.area_under_curve(h_scores, interpolate="left")
        print("--- Hourglass score AUC: %.2f ---" % hourglass_auc)
        # Randomization to test robustness against random network
        if len(randomized_connectomes) > 0:
            randomized_result = _map_wrapper(
                functools.partial(
                    _calculate_hourglass, paths=paths, weighting_str=weighting_method
                ),
                randomized_connectomes,
                multi_process=multi_process,
            )
            randomized_scores, randomized_aucs = list(
                map(list, zip(*randomized_result))
            )
            print("--- Randomization test ---")
            summary.test_value_distribution(
                randomized_aucs,
                hourglass_auc,
                value_name="H-score AUC",
                path=figure_prefix
                + "%s-hscoreauc-dist-compare.png" % data.get_abbr(weighting_method),
            )
    # Get cumulative path coverage from core nodes
    core_cum_coverage = dict()
    for weighting_method, core_nodes in agg_core_nodes.items():
        node_names, coverages = list(zip(*core_nodes.items()))
        cum_coverages = hourglass.cumsum(coverages)
        core_cum_coverage[weighting_method] = list(
            zip(range(len(core_nodes)), cum_coverages)
        )
    summary.compare_core_nodes(
        agg_core_nodes,
        xlabel="Method",
        ylabel="Neuron",
        path=figure_prefix + "core-nodes.png",
    )
    summary.multi_lineplot(
        core_cum_coverage,
        xlabel=r"Node Sequence",
        ylabel="Path Coverage",
        path=figure_prefix + "core-nodes-coverage.png",
    )
    summary.multi_lineplot(
        agg_hourglass_scores,
        xlabel=r"Path Covearge Threshold ($\tau$)",
        ylabel="H-score",
        path=figure_prefix + "hsore-lines.png",
    )


def compare_hourglass(
    properties, result_prefix, xlabel="Method", styles=None, print_table=False
):
    """ Compare the hourglass effects using different connectomes or 
    different methods. Plot the core nodes, core node path centrality coverage
    and hourglass score obtained.
    Args:
        properties (list): list of property that contains the information about
            the connectome and method to use. The property is in the format of 
            "(sex)-(edge type)-(weight type)-(path selection method)-
            (path weighting method)" to run hourglass analysis on the new 
            Cook connectome or "varshney-(path selection method)" to run on the
            old Varshney connectome. You may also add ":(name)" at the end of
            the string to specify the name wanted for this combination of 
            connectome and method in displaying the results.
        styles (list): style coding of each trial.
        result_prefix (str): prefix for the resulted figure generated.
    """
    assert isinstance(properties, list)
    if styles is None:
        styles = [None for _ in properties]
    assert isinstance(styles, list)

    agg_core_nodes = dict()
    agg_hourglass_scores = dict()
    named_styles = dict()

    for property_str, style in zip(properties, styles):
        property_name = None
        if len(property_str.split(":")) == 2:
            property_str, property_name = property_str.split(":")

        if property_str.startswith("varshney"):
            path_select_method = property_str.split("-")[-1]
            connectome = data.extract_connectome_varshney2011()
            S, I, M = data.get_sim_neurons(connectome)
            paths = path_selection.parse_select(connectome, S, M, path_select_method)
            core_nodes = hourglass.find_tau_core(paths, tau=0.9, weighted=False)
            h_scores = hourglass.hourglass_scores(
                paths, max_tau=MAX_TAU, weighted=False
            )

        else:
            (
                sex,
                connection_type,
                weight_type,
                path_select_method,
                path_weighting_method,
            ) = property_str.split("-")

            connectome = data.get_connectome(sex, [connection_type], [weight_type])
            connectome = data.preprocess(connectome)
            connectome = data.remove_feedback_edges(connectome)
            S, I, M = data.get_sim_neurons(connectome)
            paths = path_selection.parse_select(connectome, S, M, path_select_method)
            path_weights = path_selection.parse_weighting(
                connectome, paths, path_weighting_method
            )
            core_nodes = hourglass.find_tau_core(path_weights, tau=0.9, weighted=True)
            h_scores = hourglass.hourglass_scores(
                path_weights, max_tau=MAX_TAU, weighted=True
            )

        if property_name:
            agg_core_nodes[property_name] = core_nodes
            agg_hourglass_scores[property_name] = h_scores
            named_styles[property_name] = style
        else:
            agg_core_nodes[property_str] = core_nodes
            agg_hourglass_scores[property_str] = h_scores
            named_styles[property_str] = style

    core_cum_coverage = dict()
    for name, core_nodes in agg_core_nodes.items():
        node_names, coverages = list(zip(*core_nodes.items()))
        cum_coverages = hourglass.cumsum(coverages)
        core_cum_coverage[name] = list(zip(range(len(core_nodes)), cum_coverages))
    summary.compare_core_nodes(
        agg_core_nodes,
        xlabel=xlabel,
        ylabel="Neuron",
        path=result_prefix + "core-nodes.png",
    )

    if print_table:
        print(pd.DataFrame(agg_core_nodes).to_latex())

    summary.jaccard_plot(agg_core_nodes, path=result_prefix + "core-nodes-jaccard.png")
    summary.multi_lineplot(
        core_cum_coverage,
        xlabel=r"Node Sequence",
        ylabel="Path Coverage",
        styles=named_styles,
        path=result_prefix + "core-nodes-coverage.png",
    )
    summary.multi_lineplot(
        agg_hourglass_scores,
        xlabel=r"Path Covearge Threshold ($\tau$)",
        ylabel="H-score",
        styles=named_styles,
        path=result_prefix + "hsore-lines.png",
        sampling=True,
        num_samples=21,
        interpolate="right",
    )


def get_tau_score(scores, tau=0.9):
    taus = [t for t, _ in scores]
    return scores[bisect.bisect_left(taus, tau)][1]


def path_weighting_random_test(
    sex,
    connection_type,
    weight_type,
    path_select_method,
    path_weighting_method,
    randomize_method,
    num_random_samples=100,
    multi_process=False,
):
    """ Test the path weighting with randomization experiments
    Args:
        sex (str): sex of target connectome. Currently support ["herm", "male"].
        connection_type (str): target connection type. Currently support 
            ["chem synapse", "gap junction"].
        weight_type (str): target weight type. Currently support ["count", "size"].
        path_select_method (str): string that specified how path 
            selection should be conducted. Currently support "length:sp+ApB" 
            to apply length-based selection. This would result in including
            only paths A hops longer than shorted path between the same source
            and target, and not longer than B hops. 
        path_weighting_method (str): string that specified the path
            weighting method, currently support "met", "met-median", "met-log",
            "gm", "flow", "fidelity", "frequency".
        randomize_method (str): methods used to perform randomization of network.
            Currently support "edge-weight", "sabrin".
        num_random_samples (int): number of randomized networks generated to 
            compare the result on empirical network with.
        multi_process (bool): Whether to apply multi-processing during 
            randomization.
    """
    print(
        "Sex: %s, Connection Type: %s, Weight Type: %s, "
        % (sex, connection_type, weight_type)
        + "Path Selection Method: %s, Path Weigthing method: %s"
        % (path_select_method, path_weighting_method)
    )
    figure_prefix = os.path.join(
        "figs",
        "%s-%s-%s-" % tuple(map(data.get_abbr, (sex, connection_type, weight_type))),
    )
    # Get the connectome
    connectome = data.get_connectome(sex, [connection_type], [weight_type])
    # Preprocess
    connectome = data.preprocess(connectome)
    connectome = data.remove_feedback_edges(connectome)
    # Get paths based on the path selection
    S, I, M = data.get_sim_neurons(connectome)
    paths = path_selection.parse_select(connectome, S, M, path_select_method)
    # Get path weighting result
    path_weights = path_selection.parse_weighting(
        connectome, paths, path_weighting_method
    )
    # Hourglass scores and area under curve
    h_scores = hourglass.hourglass_scores(path_weights, max_tau=MAX_TAU, weighted=True)
    hourglass_auc = hourglass.area_under_curve(h_scores, interpolate="left")
    print("--- Hourglass score AUC: %.2f ---" % hourglass_auc)

    if num_random_samples is not None and num_random_samples >= 0:
        if randomize_method == "edge-weight":
            # Generate randomized networks from the connectome for robustness analysis
            randomized_connectomes = [
                randomize.permute_edge_weight(connectome)
                for _ in range(num_random_samples)
            ]
            # Randomization to test robustness against random network
            randomized_result = _map_wrapper(
                functools.partial(
                    _calculate_hourglass,
                    paths=paths,
                    weighting_str=path_weighting_method,
                ),
                randomized_connectomes,
                multi_process=multi_process,
            )
            randomized_scores, randomized_aucs = list(
                map(list, zip(*randomized_result))
            )
        elif randomize_method == "sabrin":
            randomized_connectomes = [
                randomize.randomize_sabrin(connectome, paths=paths, weighted=True)
                for _ in range(num_random_samples)
            ]
            randomized_paths = _map_wrapper(
                functools.partial(
                    path_selection.parse_select,
                    sources=S,
                    targets=M,
                    method_str=path_select_method,
                ),
                randomized_connectomes,
                multi_process=multi_process,
            )
            # Randomization to test robustness against random network
            randomized_result = _map_wrapper(
                functools.partial(
                    _calculate_hourglass, weighting_str=path_weighting_method,
                ),
                list(zip(randomized_connectomes, randomized_paths)),
                multi_args=True,
                multi_process=multi_process,
            )
            randomized_scores, randomized_aucs = list(
                map(list, zip(*randomized_result))
            )

        print("--- Randomization test ---")
        summary.test_value_distribution(
            [get_tau_score(scores) for scores in randomized_scores],
            get_tau_score(h_scores),
            value_name=r"H-score ($\tau$=0.9)",
            path=figure_prefix
            + "%s-%s-test.png"
            % (data.get_abbr(path_weighting_method), randomize_method),
        )


if __name__ == "__main__":

    # Using the existing functions in the main.py we can conduct several
    # Analysis

    ### Example 1 - Hourglass Effect Comparison ###
    # Compare the tau-core, node coverage and the hourglass scores, among
    # different configuration of datasets and methods.
    print("Example 1 - Hourglass Effect Comparison")
    compare_hourglass(
        [
            # varshney dataset with unweighted method
            "varshney-length,p4:Varshney-UNW",
            # cook dataset (hermaphrodite, chemical synapse only,
            # synapse count as weight) with unweighted method
            "herm-chem synapse-count-length,p4-uniform:Cook-UNW",
            # cook dataset (hermaphrodite, chemical synapse only,
            # synapse count as weight) with MET method
            "herm-chem synapse-count-length,p4-met:Cook-MET",
        ],
        "figs/varshney-vs-cook-",  # prefix to the resulted figure.
        xlabel="Connectome-Method",
    )

    ### Example 2 - Randomization test ###
    # Compare the hourglass on the empricial connectome and randomized
    # networks.
    
    print("Example 2 - Randomization test")
    path_weighting_random_test(
        "herm",  # hermaphrodite
        "chem synapse",  # chemical synapse only
        "count",  # synapse count as weight
        "length,p4",  # path with length <= 4
        "met",  # MET method
        randomize_method="edge-weight",  # Edge weight permutation
        num_random_samples=500,
        multi_process=True,
    )

    # For more experiment configurations, please refer to the annotation
    # of each function.

