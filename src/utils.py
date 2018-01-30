import scipy.sparse as sparse

def save_edgelist(G, fname, weighted=False, sep=' '):
    with open(fname, 'w') as f:
        for src in G:
            for dst, weight in zip(G[src], G[src].weights):
                f.write(unicode(str(src)))
                f.write(sep)
                f.write(unicode(str(dst)))
                if weighted:
                    f.write(sep)
                    f.write(unicode(str(weight)))
                f.write('\n')

def to_adjacency_matrix(G, nodelist=None, dtype=None, real=False):
    """Inspired by the NetworkX equivalent."""

    if nodelist is None:
        nodelist = sorted(G.nodes())

    nodeset = set(nodelist)
    if len(nodelist) != len(nodeset):
        raise ValueError("Ambiguous ordering: `nodelist` contained duplicates.")

    nlen=len(nodelist)
    index=dict(zip(nodelist,range(nlen)))

    I = []
    J = []
    A_ij = []

    # TODO support weights
    for src in G:
        for dst, weight in zip(G[src], G[src].weights):
            try:
                I.append(index[src])
                J.append(index[dst])
                if real:
                    A_ij.append(1. * weight)
                else:
                    A_ij.append(weight)
            except KeyError:
                pass

    A = sparse.coo_matrix((A_ij, (I, J)), shape=(nlen, nlen), dtype=dtype).tocsr()

    return A
