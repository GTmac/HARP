from gensim.models.word2vec  import Word2Vec
from multiprocessing import cpu_count
import graph_coarsening
import numpy as np

_WORKERS = 1 + int(cpu_count() / 2)  # At least half of the available logical cores

def skipgram_baseline(graph, **kwargs):
    scale = kwargs.get('scale', -1)
    representation_size = kwargs.get('representation_size', 128)

    if scale == 1:
        edges, weights = graph.get_edges()
    else:
        path_length = kwargs.get('path_length', 40)
        num_paths = kwargs.get('num_paths', 80)
        output = kwargs.get('output', 'default')
        edges = graph_coarsening.build_deepwalk_corpus(graph, num_paths, path_length, output)

    workers=kwargs.get('workers', _WORKERS)
    if kwargs['hs'] == 0:
        print ('Training the Negative Sampling Model...')
        model = Word2Vec(edges, size=representation_size, window=kwargs['window_size'], min_count=0, sg=1, hs=0, iter=kwargs['iter_count'], negative=5, workers=workers)
    else:
        print ('Training the Hierarchical Softmax Model...')
        model = Word2Vec(edges, size=kwargs['representation_size'], window=kwargs['window_size'], min_count=0, sg=1, hs=1, iter=kwargs['iter_count'], workers=workers)

    print ('Finish training the Skip-gram model.')
    return model
