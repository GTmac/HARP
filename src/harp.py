import magicgraph
import logging
import os
import sys

import numpy as np

from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from magicgraph import WeightedDiGraph, WeightedNode
from scipy.io import mmread, mmwrite, loadmat

import graph_coarsening

def main():
    parser = ArgumentParser('harp',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--format', default='mat',
                        help='File format of input file')
    parser.add_argument('--input', nargs='?', required=True,
                        help='Input graph file')
    parser.add_argument('--sfdp-path', default='./bin/sfdp_osx',
                        help='Path to the SFDP binary file which produces graph coarsening results.')
    parser.add_argument('--model', default='deepwalk',
                        help='Embedding model to use. Could be deepwalk, line or node2vec.')
    parser.add_argument('--matfile-variable-name', default='network',
                        help='Variable name of adjacency matrix inside a .mat file')
    parser.add_argument('--number-walks', default=40, type=int,
                        help='Number of random walks to start at each node')
    parser.add_argument('--output', required=True,
                        help='Output representation file')
    parser.add_argument('--representation-size', default=128, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--walk-length', default=10, type=int,
                        help='Length of the random walk started at each node.')
    parser.add_argument('--window-size', default=10, type=int,
                        help='Window size of the Skip-gram model.')
    parser.add_argument('--workers', default=1, type=int,
                        help='Number of parallel processes.')
    args = parser.parse_args()

    # Process args
    if args.format == 'mat':
        G = magicgraph.load_matfile(args.input, variable_name=args.matfile_variable_name, undirected=True)
    elif args.format == 'adjlist':
        G = magicgraph.load_adjacencylist(args.input, undirected=True)
    elif args.format == 'edgelist':
        G = magicgraph.load_edgelist(args.input, undirected=True)
    else:
        raise Exception("Unknown file format: '%s'. Valid formats: 'mat', 'adjlist', and 'edgelist'."
                % args.format)
    G = graph_coarsening.DoubleWeightedDiGraph(G)
    print ('Number of nodes: {}'.format(G.number_of_nodes()))
    print ('Number of edges: {}'.format(G.number_of_edges()))
    print ('Underlying network embedding model: {}'.format(args.model))

    if args.model == 'deepwalk':
        embeddings = graph_coarsening.skipgram_coarsening_disconnected(G,scale=-1,iter_count=1,
                sfdp_path=args.sfdp_path,
                num_paths=args.number_walks,path_length=args.walk_length,
                representation_size=args.representation_size,window_size=args.window_size,
                lr_scheme='default',alpha=0.025,min_alpha=0.001,sg=1,hs=1,coarsening_scheme=2, sample=0.1)
    elif args.model == 'node2vec':
        embeddings = graph_coarsening.skipgram_coarsening_disconnected(G,scale=-1,iter_count=1,
                sfdp_path=args.sfdp_path,
                num_paths=args.number_walks,path_length=args.walk_length,
                representation_size=args.representation_size,window_size=args.window_size,
                lr_scheme='default',alpha=0.025,min_alpha=0.001,sg=1,hs=0,coarsening_scheme=2, sample=0.1)
    elif args.model == 'line':
        embeddings = graph_coarsening.skipgram_coarsening_disconnected(G,scale=1, iter_count=50,
                sfdp_path=args.sfdp_path,
                representation_size=64,window_size=1,
                lr_scheme='default',alpha=0.025,min_alpha=0.001,sg=1,hs=0,sample=0.001)
    np.save(args.output, embeddings)

if __name__ == '__main__':
    sys.exit(main())
