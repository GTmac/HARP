import copy
import glob
import logging
import magicgraph
import math
import operator
import os
import random
import skipgram
import subprocess
import sys
import tempfile
import baseline
import utils
import numpy as np

from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from deepwalk import walks as serialized_walks
from gensim.models import Word2Vec
from magicgraph import WeightedDiGraph, WeightedNode
from scipy.io import mmread, mmwrite

class DoubleWeightedDiGraph(WeightedDiGraph):
    def __init__(self, init_graph = None):
        super(WeightedDiGraph, self).__init__(node_class=WeightedNode)
        self.weighted_nodes = magicgraph.WeightedNode()
        if init_graph is not None:
            for node, adj_list in init_graph.adjacency_iter():
                if hasattr(adj_list, 'weights'):
                    self[node].extend(adj_list, adj_list.weights)
                else:
                    self[node].extend(adj_list, [1. for adj_node in adj_list])
            if hasattr(init_graph, 'weighted_nodes'):
                self.weighted_nodes.extend(init_graph.nodes(), init_graph.weighted_nodes.weights)
            else:
                self.weighted_nodes.extend(init_graph.nodes(), [1. for node in init_graph.nodes()])
        self.visited = {node: False for node in self.nodes()}

    def is_connected(self):
        # sys.setrecursionlimit(self.number_of_nodes())
        self.visited = {node: False for node in self.nodes()}
        if self.number_of_nodes() == 0:
            return True
        self.cur_component = []
        self.bfs(list(self.nodes())[0])
        return sum(self.visited.values()) == self.number_of_nodes()

    def get_connected_components(self):
        connected_components = []
        self.visited = {node: False for node in self.nodes()}

        for node in self.nodes():
            if self.visited[node] is False:
                self.cur_component = []
                self.bfs(node)
                connected_components.append(len(self.cur_component))
        return connected_components

    # graph coarsening need to be done on each connected component
    def get_merged_connected_components(self):
        disconnected_component, connected_components, reversed_mappings = [], [], []
        self.visited = {node: False for node in self.nodes()}
        graph_size_threshold = 100

        for node in self.nodes():
            if self.visited[node] is False:
                self.cur_component = []
                self.bfs(node)
                if len(self.cur_component) >= graph_size_threshold:
                    self.cur_component = sorted(self.cur_component)
                    index_mapping = {self.cur_component[i]: i for i in range(len(self.cur_component)) }
                    connected_components.append(self.subgraph(self.cur_component, index_mapping=index_mapping))
                    reversed_mappings.append({i: self.cur_component[i] for i in range(len(self.cur_component)) })
                else:
                    disconnected_component.extend(self.cur_component)

        if len(disconnected_component) > 0:
            disconnected_component = sorted(disconnected_component)
            reversed_mappings.append({i: disconnected_component[i] for i in range(len(disconnected_component)) })
            index_mapping = {disconnected_component[i]: i for i in range(len(disconnected_component)) }
            connected_components.append(self.subgraph(disconnected_component, index_mapping=index_mapping) )
        return connected_components, reversed_mappings

    def dfs(self, cur_node):
        self.visited[cur_node] = True
        self.cur_component.append(cur_node)
        for adj_node in self[cur_node]:
            if self.visited[adj_node] is False:
                self.visited[adj_node] = True
                self.dfs(adj_node)

    def bfs(self, cur_node):
        q = deque()
        q.append(cur_node)
        self.visited[cur_node] = True

        while len(q) > 0:
            head = q.popleft()
            self.cur_component.append(head)
            for adj_node in self[head]:
                if not self.visited[adj_node]:
                    self.visited[adj_node] = True
                    q.append(adj_node)

    def subgraph(self, nodes = {}, index_mapping = None):
        nodes = set(nodes)
        if index_mapping is None:
            index_mapping = {node: node for node in nodes}
        sub = DoubleWeightedDiGraph(magicgraph.from_adjlist([ [index_mapping[node]] for node in nodes]))
        for node in nodes:
            for adj_node, weight in zip(self[node], self[node].weights):
                if adj_node in nodes:
                    sub[index_mapping[node]].append(index_mapping[adj_node], weight)
            if len(self[node]) == 0:
                if index_mapping:
                    sub[index_mapping[node]].append(index_mapping[node], 1.)
                else:
                    sub[node].append(node, 1.)

        node_weight_map = {node: weight for node, weight in zip(self.weighted_nodes, self.weighted_nodes.weights)}
        for node in nodes:
            sub.weighted_nodes.weights[index_mapping[node] ] = node_weight_map[node]
        return sub

    # get edges as pairs of integers
    def get_int_edges(self):
        edges, weights = [], []
        for node in self.nodes():
            for adj_node, weight in zip(self[node], self[node].weights):
                edges.append([node, adj_node])
                weights.append(weight)
        return edges, weights

    # get edges along with weights
    def get_edges(self):
        edges, weights = [], []
        for node in self.nodes():
            for adj_node, weight in zip(self[node], self[node].weights):
                edges.append([str(node), str(adj_node)])
                weights.append(weight)
        return edges, np.array(weights)
    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        G = self
        if start is not None:
            path = [start]
        else:
            path = [rand.choice(G.keys())]

        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    path.append(G[cur].choice(rand))
                else:
                    path.append(path[0])
            else:
                break
        return path

def external_collapsing(graph, merged):
    coarsened_graph = DoubleWeightedDiGraph()
    edges, weights = graph.get_int_edges()
    merged_edge_to_weight = defaultdict(float)
    node_weight = {node: weight for node, weight in zip(graph.weighted_nodes, graph.weighted_nodes.weights)}
    new_node_weights = defaultdict(float)
    for (a, b), w in zip(edges, weights):
        merged_a, merged_b = merged[a], merged[b]
        if merged_a != merged_b:
            merged_edge_to_weight[(merged_a, merged_b)] += w
    for node_pair, weight in merged_edge_to_weight.items():
        coarsened_graph[node_pair[0]].append(node_pair[1], weight)
        coarsened_graph[node_pair[1]].append(node_pair[0], weight)

    for node in coarsened_graph.nodes():
        coarsened_graph.weighted_nodes.append(node, new_node_weights[node])
    return coarsened_graph.make_consistent()

def read_coarsening_info(coarsening_file_dir):
    coarsening_files = [f for dirpath, dirnames, files in os.walk(coarsening_file_dir)
        for f in files if f.startswith('prolongation')]
    levels = -1
    recursive_merged_nodes = []
    for f in coarsening_files:
        levels = max(levels, int(f[f.rfind('_') + 1:]) )
    prev_rename, rename = {}, {}
    for level in range(levels + 1):
        # different index
        merged_from = defaultdict(list)
        merged = {}
        fp = open(os.path.normpath(coarsening_file_dir) + '/' + 'prolongation_' + str(level))
        for line in fp:
            finer_node, coarser_node = map(int, line.strip().split())
            # let index starts from 0 instead
            finer_node, coarser_node = finer_node - 1, coarser_node - 1
            if finer_node in prev_rename:
                # print coarser_node, finer_node, prev_rename[finer_node]
                merged_from[coarser_node].append(prev_rename[finer_node])
            else:
                merged_from[coarser_node].append(finer_node)
        # print merged_from

        for k in merged_from.keys():
            rename[k] = merged_from[k][0]
            for node in merged_from[k]:
                merged[node] = merged_from[k][0]
        # print merged
        recursive_merged_nodes.append(merged)
        prev_rename = rename.copy()
        rename = {}
    return recursive_merged_nodes

def external_ec_coarsening(graph, sfdp_path, coarsening_scheme=2):
    temp_dir = tempfile.mkdtemp()
    temp_fname = 'tmp.mtx'
    input_fname = os.path.join(temp_dir, temp_fname)
    mmwrite(open(os.path.join(input_fname), 'wb'), magicgraph.to_adjacency_matrix(graph))
    sfdp_abs_path = os.path.abspath(sfdp_path)
    subprocess.call('%s -g%d -v -u -Tc %s 2>x' % (sfdp_abs_path, coarsening_scheme, input_fname), shell=True, cwd=temp_dir)
    recursive_graphs, recursive_merged_nodes = [], read_coarsening_info(temp_dir)
    subprocess.call(['rm', '-r', temp_dir])
    cur_graph = graph
    iter_round = 1
    prev_node_count = graph.number_of_nodes()
    ec_done = False
    levels = len(recursive_merged_nodes)
    if levels == 0:
        return [graph], recursive_merged_nodes

    for level in range(levels):
        if iter_round == 1:
            print ('Original graph with %d nodes and %d edges' % \
            (cur_graph.number_of_nodes(), cur_graph.number_of_edges() ) )
            recursive_graphs.append(DoubleWeightedDiGraph(cur_graph))

        coarsened_graph = external_collapsing(cur_graph, recursive_merged_nodes[level])
        cur_node_count = coarsened_graph.number_of_nodes()
        print ('Coarsening Round %d:' % iter_round)
        print ('Generate coarsened graph with %d nodes and %d edges' % \
        (coarsened_graph.number_of_nodes(), coarsened_graph.number_of_edges()) )

        recursive_graphs.append(coarsened_graph)
        cur_graph = coarsened_graph
        iter_round += 1
        prev_node_count = cur_node_count

    return recursive_graphs, recursive_merged_nodes

def skipgram_coarsening_disconnected(graph, recursive_graphs=None, recursive_merged_nodes=None, **kwargs):
    print (kwargs)
    if graph.is_connected():
        print ('Connected graph.')
        subgraphs, reversed_mappings = [graph], [{node: node for node in graph.nodes()}]
    else:
        subgraphs, reversed_mappings = graph.get_merged_connected_components()
    count = 0
    scale = kwargs.get('scale', -1)
    num_paths = kwargs.get('num_paths', 40)
    path_length = kwargs.get('path_length', 10)
    representation_size = kwargs.get('representation_size', 128)
    window_size = kwargs.get('window_size', 10)
    iter_count = kwargs.get('iter_count', 1)
    lr_scheme = kwargs.get('lr_scheme', 'default')
    alpha = kwargs.get('alpha', 0.025)
    min_alpha = kwargs.get('min_alpha', 0.001)
    report_loss = kwargs.get('report_loss', False)
    hs = kwargs.get('hs', 0)
    sample = kwargs.get('sample', 1e-3)
    coarsening_scheme = kwargs.get('coarsening_scheme', 2)
    sfdp_path = kwargs.get('sfdp_path', './bin/sfdp_osx')
    embeddings = np.ndarray(shape=(graph.number_of_nodes(), representation_size), dtype=np.float32)

    for subgraph, reversed_mapping in zip(subgraphs, reversed_mappings):
        count += 1
        print ('Subgraph %d with %d nodes and %d edges' % (count, subgraph.number_of_nodes(), subgraph.number_of_edges()))

        if not subgraph.is_connected():
            gc_single_model = baseline.skipgram_baseline(subgraph,
                                        scale=scale,
                                        num_paths=num_paths,
                                        path_length=path_length,
                                        iter_count=iter_count,
                                        representation_size=representation_size,
                                        window_size=window_size,
                                        report_loss=report_loss,
                                        progress_threshold=100000,
                                        alpha=alpha,
                                        min_alpha=min_alpha,
                                        sg=1,
                                        hs=hs)
            gc_model = [gc_single_model]
        else:
            if recursive_graphs is None:
                print ('Graph Coarsening...')
                recursive_graphs, recursive_merged_nodes = external_ec_coarsening(subgraph, sfdp_path)
            iter_counts = [iter_count for _ in range(len(recursive_graphs))]
            if hs == 1:
                gc_model = skipgram_coarsening_hs(recursive_graphs, recursive_merged_nodes,
                                        scale=scale,
                                        iter=iter_counts,
                                        num_paths=num_paths,
                                        path_length=path_length,
                                        representation_size=representation_size,
                                        window_size=window_size,
                                        report_loss=report_loss,
                                        progress_threshold=100000,
                                        lr_scheme=lr_scheme,
                                        alpha=alpha,
                                        min_alpha=min_alpha,
                                        sg=1,
                                        hs=1,
                                        sample=sample)
            else:
                print ('Training negative sampling model...')
                gc_model = skipgram_coarsening_neg(recursive_graphs, recursive_merged_nodes,
                                        scale=scale,
                                        iter=iter_counts,
                                        num_paths=num_paths,
                                        path_length=path_length,
                                        representation_size=representation_size,
                                        window_size=window_size,
                                        report_loss=report_loss,
                                        progress_threshold=100000,
                                        lr_scheme=lr_scheme,
                                        alpha=alpha,
                                        min_alpha=min_alpha,
                                        sample=sample,
                                        sg=1,
                                        hs=0)

        for ind, vec in enumerate(gc_model[-1].syn0):
            real_ind = reversed_mapping[int(gc_model[-1].index2word[ind])]
            embeddings[real_ind] = vec
    return embeddings

def gen_alpha(init_alpha, recursive_graphs, iter_counts):
    edge_counts = [graph.number_of_edges() for graph in recursive_graphs]
    total_iter_count = sum([edge_count * iter_count for edge_count, iter_count in zip(edge_counts, iter_counts)])
    cur_iter_count, alpha_list = 0, []
    for edge_count, iter_count in zip(edge_counts, iter_counts):
        cur_iter_count += edge_count * iter_count
        alpha_list.append(init_alpha * 1. * cur_iter_count / total_iter_count)
    return alpha_list

def skipgram_coarsening_hs(recursive_graphs, recursive_merged_nodes, **kwargs):
    print (kwargs)
    print ('Start building Skip-gram + Hierarchical Softmax model on the coarsened graphs...')
    models = []
    original_graph = recursive_graphs[0]
    levels = len(recursive_graphs)
    alpha = kwargs.get('alpha', 0.25)
    min_alpha = kwargs.get('min_alpha', 0.25)
    tmp_alpha_list = gen_alpha(alpha, recursive_graphs, kwargs['iter'])
    lr_scheme = kwargs.get('lr_scheme', "default")
    sample = kwargs.get('sample', 1e-3)

    # learning rate schemes: "default", "constant", "global_linear", "local_linear"
    if lr_scheme == 'default':
        alpha_list = [alpha for i in range(levels)]
        min_alpha_list = [min_alpha for i in range(levels)]
    if kwargs["lr_scheme"] == 'constant':
        alpha_list = [alpha for i in range(levels)]
        min_alpha_list = [alpha for i in range(levels)]
    elif kwargs["lr_scheme"] == 'local_linear':
        alpha_list = [alpha for alpha in tmp_alpha_list]
        min_alpha_list = [min_alpha for i in range(levels)]
    elif kwargs["lr_scheme"] == 'global_linear':
        alpha_list = [alpha for alpha in tmp_alpha_list]
        min_alpha_list = [min_alpha]
        min_alpha_list.extend([tmp_alpha_list[i] for i in range(levels - 1)])

    scale = kwargs.get('scale', 1)
    if 'walks' in kwargs:
        walks = kwargs['walks']

    for level in range(levels - 1, -1, -1):
        print ('Training on graph level %d...' % level)
        if scale == 1:
            edges, weights = recursive_graphs[level].get_edges()
            random.shuffle(edges)
        elif scale == -1:
            path_length = kwargs.get('path_length', 10)
            num_paths = kwargs.get('num_paths', 40)
            output = kwargs.get('output', 'default')
            edges = build_deepwalk_corpus(recursive_graphs[level], num_paths, path_length, output)

        # the coarest level
        if level == levels - 1:
            model = skipgram.Word2Vec_hs_loss(edges, sg=kwargs['sg'], size=kwargs['representation_size'], iter=kwargs['iter'][level], window=kwargs['window_size'], sample=sample, alpha=alpha_list[level], min_alpha=min_alpha_list[level])
        else:
            model = skipgram.Word2Vec_hs_loss(None, sg=kwargs['sg'], size=kwargs['representation_size'], iter=kwargs['iter'][level], window=kwargs['window_size'], sample=sample, alpha=alpha_list[level], min_alpha=min_alpha_list[level])

            # copy vocab / index2word from the coarser graph
            model.vocab = copy.deepcopy(models[-1].vocab)
            model.index2word = copy.deepcopy(models[-1].index2word)
            model.syn0 = copy.deepcopy(models[-1].syn0)
            model.syn0.resize(recursive_graphs[level].number_of_nodes(), kwargs['representation_size'])
            model.syn0norm = None
            model.corpus_count = len(edges)

            cur_merged_nodes = [(node, merged_node) for node, merged_node in recursive_merged_nodes[level].iteritems() if node != merged_node]
            cur_merged_nodes = sorted(cur_merged_nodes, key=operator.itemgetter(1))

            changed_merged_nodes = []
            cur_merged_node, prev_node = -1, -1
            node_pool = []
            for node, merged_node in cur_merged_nodes:
                if merged_node == cur_merged_node:
                    changed_merged_nodes.append((node, random.choice(node_pool)))
                    node_pool.append(node)
                else:
                    changed_merged_nodes.append((node, merged_node))
                    cur_merged_node = merged_node
                    node_pool = [node, merged_node]
                prev_node = node

            cur_index = len(models[-1].vocab)
            for node, merged_node in changed_merged_nodes:
                if node == merged_node:
                    continue
                str_node, str_merged_node = str(node), str(merged_node)
                word_index = model.vocab[str_merged_node].index
                init_vec = model.syn0[word_index]
                model.add_word(str_node, str_merged_node, init_vec, cur_index)
                cur_index += 1
                model.add_word(str_merged_node, str_merged_node, init_vec, cur_index)

            model.syn1 = np.zeros((len(model.vocab), model.layer1_size), dtype=np.float32)
            for i in range(len(models[-1].syn1)):
                model.syn1[i] = models[-1].syn1[i]
            model.syn0_lockf = np.ones(len(model.vocab), dtype=np.float32)
            model.train(edges)

        models.append(model)

    print ('Finish building Skip-gram model on the coarsened graphs.')
    return models

def skipgram_coarsening_neg(recursive_graphs, recursive_merged_nodes, **kwargs):
    # print (kwargs)
    print ('Start building Skip-gram + Negative Sampling model on the coarsened graphs...')
    models = []
    original_graph = recursive_graphs[0]
    levels = len(recursive_graphs)
    tmp_alpha_list = gen_alpha(kwargs.get('alpha', 0.025), recursive_graphs, kwargs['iter'])
    # learning rate schemes: "constant", "global_linear", "local_linear"
    if kwargs["lr_scheme"] == 'default':
        alpha_list = [kwargs['alpha'] for i in range(levels)]
        min_alpha_list = [kwargs['min_alpha'] for i in range(levels)]
    if kwargs["lr_scheme"] == 'constant':
        alpha_list = [kwargs['alpha'] for i in range(levels)]
        min_alpha_list = [kwargs['alpha'] for i in range(levels)]
    elif kwargs["lr_scheme"] == 'local_linear':
        alpha_list = [alpha for alpha in tmp_alpha_list]
        min_alpha_list = [kwargs['min_alpha'] for i in range(levels)]
    elif kwargs["lr_scheme"] == 'global_linear':
        alpha_list = [alpha for alpha in tmp_alpha_list]
        min_alpha_list = [kwargs['min_alpha']]
        min_alpha_list.extend([tmp_alpha_list[i] for i in range(levels - 1)])
    scale = kwargs.get('scale', 1)
    sample = kwargs.get('sample', 1e-3)

    for level in range(levels - 1, -1, -1):
        print ('Training on graph level %d...' % level)
        # DeepWalk
        if scale == -1:
            path_length = kwargs.get('path_length', 10)
            num_paths = kwargs.get('num_paths', 40)
            output = kwargs.get('output', 'default')
            edges = build_deepwalk_corpus(recursive_graphs[level], num_paths, path_length, output)
        # use adjacency matrix
        elif scale == 1:
            edges, weights = recursive_graphs[level].get_edges()
            random.shuffle(edges)

        # the coarest level
        if level == levels - 1:
            model = Word2Vec(edges, size=kwargs['representation_size'], window=kwargs['window_size'], min_count=0, sample=sample, sg=1, hs=0, iter=kwargs['iter'][level], workers=20)
        else:
            model = Word2Vec(None, size=kwargs['representation_size'], window=kwargs['window_size'], min_count=0, sample=sample, sg=1, hs=0, iter=kwargs['iter'][level], workers=20)
            model.build_vocab(edges)
            model.reset_weights()

            # init model weights with the previous one
            prev_syn0 = {models[-1].index2word[ind]: vec for ind, vec in enumerate(models[-1].syn0)}
            prev_syn1neg = {models[-1].index2word[ind]: vec for ind, vec in enumerate(models[-1].syn1neg)}
            word2index = {model.index2word[ind]: ind for ind in range(recursive_graphs[level].number_of_nodes())}
            for ind in range(recursive_graphs[level].number_of_nodes()):
                word = model.index2word[ind]
                if word in prev_syn0:
                    model.syn0[ind] = prev_syn0[word]
                    model.syn1neg[ind] = prev_syn1neg[word]
                else:
                    # if a is merged into b, then a should has identical weights in word2vec as b
                    if int(word) in recursive_merged_nodes[level]:
                        word_ind = word2index[word]
                        merged_word = str(recursive_merged_nodes[level][int(word)])
                        model.syn0[word_ind] = prev_syn0[merged_word]
                        model.syn1neg[word_ind] = prev_syn1neg[merged_word]
            model.syn0_lockf = np.ones(len(model.vocab), dtype=np.float32)

            model.train(edges)

        models.append(model)

    print ('Finish building Skip-gram model on the coarsened graphs.')
    return models

class combine_files_iter:
    def __init__(self, file_list, length, path_length):
        self.file_list = file_list
        self.file_list_iter = iter(file_list)
        self.fp_iter = open(next(self.file_list_iter))
        self.length = length
        self.path_length = path_length

    def __len__(self):
        return self.length

    def __iter__(self):
        for fname in self.file_list:
            for line in open(fname):
                yield line.split()
        # return self

    def next(self):
        try:
            result = next(self.fp_iter).split()
        except:
            try:
                self.fp_iter.close()
                self.fp_iter = open(next(self.file_list_iter))
                result = next(self.fp_iter).split()
            except:
                raise StopIteration
        return result

def build_deepwalk_corpus(G, num_paths, path_length, output, alpha=0):
    walks_filebase = output + '.walks'
    walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=num_paths,
                                         path_length=path_length, alpha=alpha, rand=random.Random(random.randint(0, 2**31)),
                                         num_workers=20)
    return combine_files_iter(walk_files, G.number_of_nodes() * num_paths, path_length)
