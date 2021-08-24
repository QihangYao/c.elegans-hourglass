# Weighted-Hourglass

Code repository for the paper "[A Weighted Network Analysis Framework for the Hourglass Effect - and its Application in the C. Elegans Connectome](https://www.biorxiv.org/content/10.1101/2021.03.19.436224v1)" by Ishaan Batta, Qihang Yao, Kaeser M. Sabrin, and Constantine Dovrolis.



## Code Structure

- `/data.py` -- Functions for building, reading and preprocessing connectome data.
- `/hourglass.py` -- Functions for hourglass analysis based on paths.
- `/main.py` -- Main script and functions for hourglass analysis on weighted C.elegans connectome.
- `/path_selection.py` -- PathSelection class, which mainly contains functions for selection and weighting of paths in a connectome.
- `/randomize.py` -- Functions for randomization of connectomes.
- `/summary.py` -- Functions to summarize the data, intermediate result and final result. 
- `/toy_examples.py` -- Provide toy example classes which we can test PathSelection on.


## Prerequisite

### Data Source

This project concerns two open-source datasets:
1. WormAtlas dataset: [Publication](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001066), [Dataset](https://www.wormatlas.org/neuronalwiring.html#NeuronalconnectivityII)
2. WormWiring dataset: [Publication](https://www.nature.com/articles/s41586-019-1352-7), [Dataset](https://wormwiring.org/)

Currently in `data.py` there are several places where we hard-coded the paths to be `data/Varshney2011` and `data/Cook2019` for these two datasets, correspondingly. However, you can still replace the hard-coded path with the paths you like. 

### Dependencies

This code is developed and tested based on `Python 3.7`. See `requirements.txt` for the dependencies. If you have `pip` installed, you can use `pip install -r requirements.txt` to install all the neccessary libraries.

## Usage

See `main.py` for several analysis pipelines that are already composed in the form of functions, and the examples provided at the end of the file. You can rely on the docstring of those functions to change parameters of the analysis.
