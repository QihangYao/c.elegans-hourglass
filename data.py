""" Functions for building, reading and preprocessing connectome data. """
import os
import urllib.request
import urllib.parse

import networkx as nx
import pandas as pd


CONNECTION_TYPES = ["chem synapse", "gap jn"]
WEIGHT_TYPES = ["count", "size"]
ABBR = {"chem synapse": "cs", "gap junction": "gj", "count": "ct", "size": "sz"}

DATA_DIR = "data"
COOK_DATASET_DIR = os.path.join("data", "Cook2019")
VARSHNEY_DATASET_DIR = os.path.join("data", "Varshney2011")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(VARSHNEY_DATASET_DIR):
    os.makedirs(VARSHNEY_DATASET_DIR)
    print("Download the Varshney2011 dataset...")
    file_urls = [
        (name, "https://www.wormatlas.org/images/" + name)
        for name in ["NeuronConnect.xls"]
    ]
    for name, url in file_urls:
        urllib.request.urlretrieve(url, os.path.join(VARSHNEY_DATASET_DIR, name))

if not os.path.exists(COOK_DATASET_DIR):
    os.makedirs(COOK_DATASET_DIR)
    print("Download the Cook2019 dataset...")
    file_urls = [
        (name, "https://wormwiring.org/si/" + urllib.parse.quote(name))
        for name in [
            "SI 2 Synapse adjacency matrices.xlsx",
            "SI 4 Cell lists.xlsx",
            "SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",
        ]
    ]
    for name, url in file_urls:
        urllib.request.urlretrieve(url, os.path.join(COOK_DATASET_DIR, name))


def get_abbr(name):
    """ Return the abbreviation of a name if exists. """
    if name in ABBR:
        return ABBR[name]
    else:
        return name


def extract_cell_list(sex, dataset_path, verbose=False):
    """ Extract cell list of the Cook 2019 dataset.
    Args:
        sex (str): sex of target connectome. Currently support ["herm", "male"].
        dataset_path (str): path to the Cook 2019 connectome dataset.
    Returns:
        cell_list (pandas.DataFrame): Cell list containing information like
            cell name, cell type, cell category.
    """
    # Load the cell list
    cell_list_path = os.path.join(dataset_path, "SI 4 Cell lists.xlsx")

    cell_list = pd.read_excel(cell_list_path, sheet_name="sex-shared").rename(
        columns={"Unnamed: 0": "cell name", "Unnamed: 4": "comment"}
    )

    pharynx_cell_list = pd.read_excel(
        cell_list_path,
        sheet_name="pharynx",
        header=None,
        names=["cell name", "cell type"],
    )
    pharynx_cell_list["cell category"] = "pharynx"
    if sex == "herm":
        sex_specific_cell_list = pd.read_excel(
            cell_list_path,
            sheet_name="hermaphrodite specific",
            header=None,
            index_col=False,
            names=["cell name", "cell type", "cell category"],
        )

    elif sex == "male":
        sex_specific_cell_list = pd.read_excel(
            cell_list_path, sheet_name="male-specific",
        ).rename(columns={"name": "cell name"})
        sex_specific_cell_list.loc[
            sex_specific_cell_list["cell type"].isin(["sensory neuron", "interneuron"]),
            "cell category",
        ] = "sex-specific neuron"

    # Merge
    cell_list = pd.concat(
        [cell_list, pharynx_cell_list, sex_specific_cell_list]
    ).reset_index(drop=True)

    cell_list["cell type"] = cell_list["cell type"].replace(
        {"sensory": "sensory neuron"}
    )

    if verbose:
        print(cell_list.groupby("cell type").size())
        print(cell_list.groupby("cell category").size())

    return cell_list


# Load adjacency data
def read_adjacency_matrix(filename, sheet_name):
    """ Read adjacency matrix from an excel sheet.
    Args:
        filename (str): name of the excel file.
        sheet_name (str): name of the target sheet.
    """
    matrix = pd.read_excel(
        filename, sheet_name=sheet_name, skiprows=[0, 1], skipfooter=1
    )
    # Drop column head and tail
    matrix = matrix.drop(matrix.columns[[0, 1, -1]], axis=1)
    # Set index and fill NaN
    matrix = matrix.rename(columns={"Unnamed: 2": "Source\Target"}).set_index(
        "Source\Target"
    )
    return matrix


def extract_edgelist(sex, dataset_path, connection_type, weight_type):
    """ Extract an adjacency matrix from the dataset.
    Args:
        sex (str): sex of the target connectome. Currently support ["herm", "male"].
        dataset_path (str): path to Cook2019 dataset.
        connection_type (str): type of connectione. Could either be 
            "gap jn" or "chem synapse".
        weight_type (str): type of weight. Could either be "count" or "size"m
            for number of synapses between each pair of neurons or sum of size
            of synapses.
    Returns:
        edgelist (dict): Extracted edgelist (u, v) - weight.
    """
    assert (type(sex) == str) and (sex in ["herm", "male"])
    assert (type(connection_type) == str) and (
        connection_type in ["gap jn", "chem synapse"]
    )
    assert (type(weight_type) == str) and (weight_type in WEIGHT_TYPES)
    # Resolving naming differences in different files.
    if weight_type == "size":
        matrix_path = os.path.join(
            dataset_path, "SI 5 Connectome adjacency matrices, corrected July 2020.xlsx"
        )
        if sex == "herm":
            sheet_name_map = {
                "chem synapse": "hermaphrodite chemical",
                "gap jn": "hermaphrodite gap jn symmetric",
            }
        else:
            sheet_name_map = {
                "chem synapse": "male chemical",
                "gap jn": "male gap jn symmetric",
            }
    else:
        matrix_path = os.path.join(dataset_path, "SI 2 Synapse adjacency matrices.xlsx")
        if sex == "herm":
            sheet_name_map = {
                "chem synapse": "herm chem synapse adjacency",
                "gap jn": "herm gap jn synapse adjacency",
            }
        else:
            sheet_name_map = {
                "chem synapse": "male chem synapse adjacency",
                "gap jn": "male gap jn synapse adjacency",
            }

    if weight_type == "count" and sex == "male":
        adjacency_matrix = pd.read_excel(
            matrix_path, sheet_name=sheet_name_map[connection_type]
        )
        adjacency_matrix = adjacency_matrix.rename(
            columns={"Unnamed: 0": "Source\Target"}
        ).set_index("Source\Target")
    else:
        adjacency_matrix = read_adjacency_matrix(
            matrix_path, sheet_name=sheet_name_map[connection_type]
        )

    return adjacency_matrix.stack().to_dict()


def compare_cell_list_connectome(cell_list, connectome):
    """ Compare the cells present in cell list and connectome, to reveal 
    possible discrepancy between the data.
    Args:
        cell_list (Pandas.DataFrame): extracted cell list from the dataset.
        connectome (NetworkX.Graph): extracted connectome from the dataset.
    """
    # Compare between cell list and connectome that we have
    print("Node appearing in the connectome but not cell list:")
    print(set(connectome.nodes()).difference(cell_list["cell name"].values))
    print("Node appearing in the cell list but not connectome:")
    print(set(cell_list["cell name"].values).difference(connectome.nodes()))

    print("Neurons appearing in the cell list but not connectome:")
    print(
        set(
            cell_list[
                cell_list["cell type"].isin(
                    ["sensory neuron", "interneuron", "motorneuron"]
                )
            ]["cell name"].values
        ).difference(connectome.nodes())
    )


def extract_connectome_cook2019(
    sex,
    connection_types,
    weight_types,
    graph_type,
    dataset_path=COOK_DATASET_DIR,
    verbose=False,
):
    """ Extract the connectome as a graph from the Cook 2019 dataset.
    Args:
        sex (str): sex of the connectome to be extracted. 'herm' or 'male'
        connection_types (list): type of the connections to be included in graph.
        weight_types (list): type of connection weights to be included in graph.
        graph_type (str): type of graph. "graph" for single-edge graph or 
            "multi-graph" for multi-edge graph. If "multi-graph" is specified,
            multiple edge is allowed between two neurons with the connection type
            as key to discriminate between the edges.
        dataset_path (str): path to the Cook2019 connectome dataset.
    Returns:
        G (NetworkX.DiGraph): NetworkX DiGraph instances used to represent the
            extracted connectome.
    """
    # Check parameters
    assert isinstance(sex, str) and (sex in ["herm", "male"])
    assert isinstance(connection_types, list) and any(
        [v in CONNECTION_TYPES for v in connection_types]
    )
    assert isinstance(weight_types, list) and any(
        [v in WEIGHT_TYPES for v in weight_types]
    )
    assert isinstance(graph_type, str) and (graph_type in ["graph", "multi-graph"])
    assert isinstance(dataset_path, str) and os.path.exists(dataset_path)

    # Get cell list
    cell_list = extract_cell_list(sex, dataset_path, verbose=verbose)
    # Get edge list for different types of connections and weights.
    edge_list = dict()
    for c_type in connection_types:
        for w_type in weight_types:
            edge_list[(c_type, w_type)] = extract_edgelist(
                sex, dataset_path, c_type, w_type
            )

    # Create a overall edge list
    weighted_edge_list = {}
    for (c_type, w_type), e_list in edge_list.items():
        for e, v in e_list.items():
            if not e in weighted_edge_list:
                weighted_edge_list[e] = {}
            weighted_edge_list[e][" ".join((c_type, w_type))] = v
    # Write weighted edge list to lines and parse using NetworkX framework
    lines = []
    for key, value in weighted_edge_list.items():
        line = str(key[0]) + " " + str(key[1]) + " " + str(value)
        lines.append(line)
    connectome = nx.parse_edgelist(lines, create_using=nx.DiGraph)

    # Create a multi-edge version (discriminate chemical synapse and gap junctions)
    if graph_type == "multi-graph":
        connectome_multi = nx.MultiDiGraph()
        for u, v, attrs in connectome.edges(data=True):
            for c_type in connection_types:
                type_attrs = [attr for attr in attrs if c_type in attr]
                # Add edge specific to this type of connection
                connectome_multi.add_edge(
                    u, v, key=c_type, **{a: attrs[a] for a in type_attrs}
                )
        connectome = connectome_multi

    # Compare the cell list and cells present in connectome
    if verbose:
        compare_cell_list_connectome(cell_list, connectome)

    # Adding the neuron information to the node
    cell_information = (
        cell_list[["cell name", "cell type", "cell category"]]
        .set_index("cell name")
        .to_dict(orient="index")
    )
    # NOTE: There are some naming differences. However, it might not affect
    # further analysis (as they are mainly endorgan nodes or muscles)
    nx.set_node_attributes(connectome, cell_information)
    # Check if all the nodes have appeared in the connectome (therefore having
    # connections)

    return connectome


def extract_connectome_varshney2011(dataset_path=VARSHNEY_DATASET_DIR):
    """ Extract connectome as a graph from the Varshney 2011 dataset.
    Args:
        dataset_path (str): path to the Varshney2011 connectome dataset.
    Returns:
        G (networkX.DiGraph): NetworkX DiGraph instances used to represent the
            extracted connectomes.
    """
    # Use the cell list in the Cook dataset for simplicity
    cell_list = extract_cell_list("herm", COOK_DATASET_DIR)
    # Get edge list
    edge_list = pd.read_excel(os.path.join(dataset_path, "NeuronConnect.xls"))
    # Use only the chemical synapses
    edge_list = edge_list[edge_list["Type"].isin(["S", "Sp"])]
    # Transform into unique dictionary
    edge_list = edge_list.groupby(["Neuron 1", "Neuron 2"]).size().to_dict()
    # Create the directed graph
    G = nx.DiGraph()
    G.add_edges_from(edge_list.keys())
    # Assign node type property
    cell_information = (
        cell_list[["cell name", "cell type", "cell category"]]
        .set_index("cell name")
        .to_dict(orient="index")
    )
    nx.set_node_attributes(G, cell_information)

    return G


def get_connectome(
    sex, connection_types=["chem synapse"], weight_types=["count"], graph_type="graph",
):
    """ Get the connectome either through extraction or read caches. 
    Args:
        sex (str): sex of the connectome to be extracted. Could either be
            "herm" or "male".
    Returns:
        connectome (NetworkX.DiGraph): extracted connectome.
    """
    # Check parameters
    assert isinstance(sex, str) and (sex in ["herm", "male"])
    assert isinstance(connection_types, list) and any(
        [v in CONNECTION_TYPES for v in connection_types]
    )
    assert isinstance(weight_types, list) and any(
        [v in WEIGHT_TYPES for v in weight_types]
    )
    assert isinstance(graph_type, str) and (graph_type in ["graph", "multi-graph"])
    # Create connectome filename based on type
    type_str = "_".join([ABBR[c] for c in (connection_types + weight_types)])
    connectome_path = "data/%s_%s_" % (sex, type_str)
    if graph_type == "multi-graph":
        connectome_path += "multi_"
    connectome_path += "connectome.graphml"
    # Create cache if not exist
    if not os.path.exists(connectome_path):
        print("Creating the cache for the Cook2019 connectome.")
        connectome = extract_connectome_cook2019(
            sex, connection_types, weight_types, graph_type
        )
        nx.write_graphml(connectome, connectome_path)
    # use cache if exist
    else:
        connectome = nx.read_graphml(connectome_path)
    return connectome


def to_edge_list(connectome, directory):
    """ Get edge list, source list, target list from the connectome. """
    edge_list = connectome.edges()
    S, _, M = get_sim_neurons(connectome)
    with open(os.path.join(directory, "edge_list"), "w") as f:
        for u, v in edge_list:
            f.write("%s %s\n" % (u, v))
    with open(os.path.join(directory, "source_list"), "w") as f:
        for s in S:
            f.write("%s\n" % (s))
    with open(os.path.join(directory, "target_list"), "w") as f:
        for m in M:
            f.write("%s\n" % (m))
    return


def preprocess(
    connectome, weight="chem synapse count", remove_sex_specific=False, verbose=False
):
    """ Preprocessing of the connectome data."""
    if verbose:
        print("--- Preprocessing the connectome ---")
    neurons = {n: data for n, data in connectome.nodes(data=True)}
    if verbose:
        print("The connectome initially has %d nodes" % (len(neurons)))
    # Remove pharynx nodes
    neurons = {
        n: data
        for n, data in neurons.items()
        if not (("cell category" in data) and (data["cell category"] == "pharynx"))
    }
    if verbose:
        print("After removing pharynx, it has %d nodes" % (len(neurons)))
    # Filtering based on the cell types
    neurons = {
        n: data
        for n, data in neurons.items()
        if ("cell type" in data)
        and (data["cell type"] in ["sensory neuron", "interneuron", "motorneuron"])
    }
    if verbose:
        print("After selecting S/I/M neurons, it has %d nodes" % (len(neurons)))
    # Get the subgraph
    connectome = connectome.subgraph(neurons.keys())
    # Are they weakly connected?
    if not nx.is_weakly_connected(connectome):
        if verbose:
            print("The filtered connectome is not weakly connected.")
        components = list(nx.weakly_connected_components(connectome))
        connectome = connectome.subgraph(components[0])
        for i, c in enumerate(components[1:]):
            if verbose:
                print("Component #%d" % (i + 1))
                for n in c:
                    print(n)
    # Remove sex specific neurons if required.
    if remove_sex_specific:
        shared_neurons = [
            n
            for n, data in connectome.nodes(data=True)
            if not (
                ("cell category" in data)
                and (data["cell category"] == "sex-specific neuron")
            )
        ]
        connectome = connectome.subgraph(shared_neurons)
    # Assign chem synapse as path weight
    for u, v, data in connectome.edges(data=True):
        connectome.edges[(u, v)]["weight"] = connectome.edges[(u, v)][weight]

    return connectome


def get_sim_neurons(connectome):
    cell_types = {n: data["cell type"] for n, data in connectome.nodes(data=True)}
    return [
        [n for n, t in cell_types.items() if t == cell_type]
        for cell_type in ["sensory neuron", "interneuron", "motorneuron"]
    ]


def remove_feedback_edges(connectome):
    """ Remove the feedback edges from the connectome. """
    edges_to_remove = []
    for u, v in connectome.edges():
        if connectome.nodes[u]["cell type"] == "interneuron":
            if connectome.nodes[v]["cell type"] == "sensory neuron":
                edges_to_remove.append((u, v))
        elif connectome.nodes[u]["cell type"] == "motorneuron":
            if connectome.nodes[v]["cell type"] in ["sensory neuron", "interneuron"]:
                edges_to_remove.append((u, v))
    ff_connectome = connectome.copy()
    ff_connectome.remove_edges_from(edges_to_remove)
    return ff_connectome

