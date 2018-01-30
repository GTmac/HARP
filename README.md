# HARP
Code for the AAAI 2018 paper "HARP: Hierarchical Representation Learning for Networks".
HARP is a meta-strategy to improve several state-of-the-art network embedding algorithms, such as *DeepWalk*, *LINE* and *Node2vec*.

You can read the preprint of our paper on [Arxiv](https://arxiv.org/abs/1706.07845).

This code run with Python 2.

# Installation

The following Python packages are required to install HARP.

[magicgraph](https://github.com/phanein/magic-graph) is a library for processing graph data.
To install, run the following commands:

	git clone https://github.com/phanein/magic-graph.git
	cd magic-graph
	python setup.py install

Then, install HARP and the other requirements:

	git clone https://github.com/GTmac/HARP.git
	cd HARP
	pip install -r requirements.txt

# Usage
To run HARP on the *CiteSeer* dataset using *LINE* as the underlying network embedding model, run the following command:

``python src/harp.py --input example_graphs/citeseer/citeseer.mat --model line --output citeseer.npy --sfdp-path bin/sfdp_linux``

Parameters available:

**--input:** *input_filename*
1. ``--format mat`` for a Matlab .mat file containing an adjacency matrix.
By default, the variable name of the adjacency matrix is ``network``;
you can also specify it with ``--matfile-variable-name``.
2. ``--format adjlist`` for an adjacency list, e.g:

	``1 2 3 4 5 6 7 8 9 11 12 13 14 18 20 22 32``
	
	``2 1 3 4 8 14 18 20 22 31``
	
	``3 1 2 4 8 9 10 14 28 29 33``
	
	``...``

3. ``--format edgelist`` for an edge list, e.g:

	``1 2``
	
	``1 3``
	
	``1 4``
	
	``2 5``
	
	``...``

**--output:** *output_filename*
The output representations in Numpy ``.npy`` format.
Note that we assume the nodes in your input file are indexed **from 0 to N - 1**.

**--model** *model_name*
The underlying network embeddings model to use. Could be ``deepwalk``, ``line`` or ``node2vec``.
Note that ``node2vec`` uses the default parameters, which is p=1.0 and q=1.0.

**--sfdp-path** *sfdp_path*
Path to the binary file of SFDP, which is the module we used for graph coarsening.
You can set it to either ``sfdp_linux`` or ``sfdp_osx`` depending on your operating system.

**More options:** The full list of command line options is available with ``python src/harp.py --help``.

# Evaluation
To evaluate the embeddings on a multi-label classification task, run the following command:

``python src/scoring.py -e citeseer.npy -i example_graphs/citeseer/citeseer.mat -t 1 2 3 4 5 6 7 8 9``

Where ``-e`` specifies the embeddings file, ``-i`` specifies the ``.mat`` file containing node labels,
and ``-t`` specifies the list of training example ratios to use.

# Note

SFDP is a library for multi-level graph drawing, which is a part of [GraphViz](http://www.graphviz.org).
We use SFDP for graph coarsening in this implementation.
Note that SFDP is included as a binary file under ``/bin``;
please choose the proper binary file according to your operation system.
Currently we have the binary files under OSX and Linux.
If you need to run HARP on a different platform, please let me know.

# Citation
If you find HARP userful in your research, please cite our paper:

	@inproceedings{harp,
		title={HARP: Hierarchical Representation Learning for Networks},
		author={Chen, Haochen and Perozzi, Bryan and Hu, Yifan and Skiena, Steven},
		booktitle={Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence},
		year={2018},
		organization={AAAI Press}
	}
