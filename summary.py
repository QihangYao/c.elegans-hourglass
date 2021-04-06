""" Functions to summarize the data, intermediate result and final result. """
import numpy as np
import pandas as pd

import scipy
import matplotlib.pyplot as plt
import seaborn as sns


def connectome_info(connectome):
    """ Show a brief summary of the connectome. 
    Arg:
        connectome (NetworkX.DiGraph): connectome in the representation
            of directed graph.
    """
    print("--- Brief Summary of the connectome ---")
    print("Number of nodes:\t%d" % len(connectome.nodes()))
    print("Number of edges:\t%d" % len(connectome.edges()))
    cell_types = [data["cell type"] for n, data in connectome.nodes(data=True)]
    cell_type_counts = {
        n: c for n, c in zip(*np.unique(cell_types, return_counts=True))
    }
    print("Sensory neurons:\t%d" % cell_type_counts["sensory neuron"])
    print("Interneurons:\t%d" % cell_type_counts["interneuron"])
    print("Motorneurons:\t%d" % cell_type_counts["motorneuron"])


def path_weight_statistics(path_weights):
    """ Show a brief summary of path weights. 
    Args:
        path_weights (dict): path weights with path as key and weight as value.
    """
    print("--- Statistics about extracted path weights ---")
    stats = scipy.stats.describe(list(path_weights.values()))
    print(
        "N: %d, Min: %.1f, Max: %.1f, Mean: %.2f, Std: %.2f"
        % (
            stats.nobs,
            stats.minmax[0],
            stats.minmax[1],
            stats.mean,
            stats.variance ** 0.5,
        )
    )
    print("Skewness: %.2f, Kurtosis: %.2f" % (stats.skewness, stats.kurtosis))
    print("--- Top 10 paths and corresponding weights ---")
    for p, v in sorted(path_weights.items(), key=lambda x: x[1], reverse=True)[:10]:
        print("%s \t %.2f" % ("-".join(p), v))


def test_value_distribution(
    reference_values, test_value, value_name=None, ax=None, path=None
):
    """ Test whether a value comes from the distribution of a set of values. 
    Args:
        reference_values (iterable): a group of values which should come
            from the same distribution.
        test_value (int/float): a value to test with the distribution.
        value_name (str): (optional) name of the value.
        ax (matplotlib.ax): (optional) if specified, plot the distribution on
            the ax.
        path (str): (optional) if specified, plot the distribution and save
            the figure to the path.
    """
    # Don't allow ax and path to be specified together
    assert not ((ax is not None) and (path is not None))
    statistic, pvalue = scipy.stats.ttest_1samp(reference_values, test_value)
    print(
        "T-statistics: %.2f, p-value: %.2f, Test-value: %.2f\n"
        "Random-value -- Mean: %.2f, Std: %.2f, Max: %.2f, Min: %.2f"
        % (
            statistic,
            pvalue,
            test_value,
            np.mean(reference_values),
            np.std(reference_values),
            np.max(reference_values),
            np.min(reference_values),
        )
    )
    if path is not None:
        _, ax = plt.subplots(figsize=(4, 3))
    if ax is not None:

        sns.histplot(
            reference_values,
            stat="probability",
            bins=40,
            kde=True,
            ax=ax,
            linewidth=0,
            # axlabel=value_name,
            label=r"Ensemble of $G_{random}$",
        )
        ax.set_xlabel(value_name)
        ax.scatter(
            [test_value],
            [ax.get_ylim()[1] / 100],
            color="green",
            alpha=0.5,
            label=r"$H-score_{real}$",
        )
        plt.ylabel("Probability")
        plt.grid()
        plt.legend()
    if path is not None:
        plt.savefig(path, bbox_inches="tight", dpi=300)


def multi_lineplot(
    values,
    xlabel=None,
    ylabel=None,
    styles=None,
    ax=None,
    path=None,
    sampling=False,
    num_samples=None,
    interpolate=None,
):
    """ Show line plot of multiple series. 
    Args:
        values (dict): dictionary of values to be plotted. Key is the name to
            discriminate between series, and value is a list of (x, y)s.
        xlabel (str): (optional) label on x axis
        ylabel (str): (optional) label on y axis.
        styles (dict): (optional) specify the style for the line.
        ax (matplotlib.ax): (optional) if specified, plot the distribution on
            the ax.
        path (str): (optional) if specified, plot the distribution and save
            the figure to the path.
        sampling (bool): whether to use sampling to smooth the curve
        num_samples (int): number of samples to take.
        interpolate (str): Way to assign value on the sampling point. 
            Support "left", "right" and "linear".
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    if sampling:
        x_range = (
            max([min(list(zip(*vs))[0]) for vs in values.values()]),
            min([max(list(zip(*vs))[0]) for vs in values.values()]),
        )
        x_ticks = np.arange(
            x_range[0], x_range[1] + 1e-3, (x_range[1] - x_range[0]) / num_samples
        )
        for name, vs in values.items():
            y_sampled = []
            for x in x_ticks:
                xs, ys = list(zip(*vs))
                if x in xs:
                    y_sampled.append(dict(vs)[x])
                else:
                    x_index = sum(x > xs)
                    if interpolate == "left":
                        y_sampled.append(ys[x_index - 1])
                    elif interpolate == "right":
                        y_sampled.append(ys[x_index])
                    elif interpolate == "linear":
                        y_sampled.append((ys[x_index - 1] + ys[x_index]) / 2)
            if styles[name] is None:
                ax.plot(x_ticks, y_sampled, label=name)
            else:
                ax.plot(x_ticks, y_sampled, styles[name], label=name)
    else:
        for name, vs in values.items():
            xs, ys = list(zip(*vs))
            if styles[name] is None:
                ax.plot(xs, ys, label=name)
            else:
                ax.plot(xs, ys, styles[name], label=name)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    plt.grid()
    plt.legend()
    if path is not None:
        plt.savefig(path, bbox_inches="tight", dpi=300)


def core_nodes_info(connectome, core_nodes):
    """ Print information about the core nodes extracted. 
    Args:
        connectome (NetworkX.DiGraph): Original connectome.
        core_nodes (dict): Extracted core nodes.
    """
    nodes = []
    for n in core_nodes:
        nodes.append(
            {
                "name": n,
                "explained ratio": core_nodes[n],
                "type": connectome.nodes[n]["cell type"],
            }
        )
    print(pd.DataFrame(nodes))


def core_nodes_path_info(core_nodes, path_weights, ax=None, path=None):
    """ Display the weight-length distribution of paths traversing the core 
    nodes. """
    path_lengths = [len(p) for p in path_weights]
    core_path_length_group = {
        i: [] for i in range(min(path_lengths), max(path_lengths) + 1)
    }
    for p, weight in path_weights.items():
        if len(set(core_nodes.keys()).intersection(p)) > 0:
            core_path_length_group[len(p)].append(weight)
    if ax is None:
        fig, ax = plt.subplots()
    ax.bar(
        range(1, len(core_path_length_group) + 1),
        [len(group) for group in core_path_length_group.values()],
        alpha=0.5,
    )
    ax.set_xlabel("Path Length Group")
    ax.set_ylabel("Count")
    ax.set_yscale("log")

    ax2 = ax.twinx()
    ax2.boxplot(core_path_length_group.values())
    ax2.set_ylabel("Path Weight")
    ax2.set_yscale("log")

    ax.set_xticklabels(core_path_length_group.keys(), rotation="vertical")

    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches="tight")


def compare_core_nodes(agg_core_nodes, ax=None, path=None, xlabel=None, ylabel=None):
    """ Compare the core nodes extracted by different methods.
    Args:
        agg_core_nodes (dict): method name as key and core nodes dict as values.
    """
    core_nodes_df = pd.DataFrame(agg_core_nodes)
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 6))
    sns.heatmap(
        core_nodes_df,
        annot=True,
        cmap="YlGn",
        ax=ax,
        cbar_kws={"label": "incremental path coverage"},
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches="tight")


def jaccard_score(a, b):
    """ Calculate the Jaccard score between two sets. 
    Args:
        a (set): a set of elements.
        b (set): another set of elements.
    Returns:
        score (float): jaccard score.
    """
    return float(len(set(a).intersection(b))) / (len(set(a).union(b)))


def jaccard_plot(agg_core_nodes, ax=None, path=None):
    """ Show the heatmap of jaccard indices between the core nodes generated
    using different methods.
    Args:
        agg_core_nodes (dict): method name as key and core nodes dict as values.
    """
    jaccard_scores = [
        [jaccard_score(a, b) for b in agg_core_nodes.values()]
        for a in agg_core_nodes.values()
    ]
    if ax is None:
        fig, ax = plt.subplots()
    sns.heatmap(
        jaccard_scores,
        annot=True,
        vmin=0,
        vmax=1,
        cmap="YlOrRd",
        xticklabels=agg_core_nodes.keys(),
        yticklabels=agg_core_nodes.keys(),
    )
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches="tight")

