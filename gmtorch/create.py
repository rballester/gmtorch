import torch
import gmtorch as gm
import itertools
import numpy as np


def create_random(nnodes=10, cardinality=2, nedges=None, seed=None):
    """
    Create a random `Graph`.

    TODO: generate higher-order edges

    :param nnodes: integer > 0
    :param cardinality: int or list of ints
    :param nedges: integer >= 0
    :param seed: for the random generator
    :return: a `Graph`
    """

    if nedges is None:
        nedges = nnodes
    if seed is None:
        seed = torch.randint(torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max, [1]).item()
    if not hasattr(cardinality, '__len__'):
        cardinality = [cardinality]*nnodes
    rng = torch.Generator()
    rng.manual_seed(seed)
    combinations = list(itertools.combinations(range(nnodes), 2))
    if nedges > len(combinations):
        raise ValueError('A network of {} nodes can have at most {} edges'.format(nnodes, len(combinations)))
    n = int(np.ceil(np.log(nnodes)/np.log(26)))
    chars = [chr(ord('A')+i) for i in range(26)]
    nodes = []
    for i in range(1, n + 1):
        nodes.extend(["".join(item) for item in itertools.product(chars, repeat=i)])
    nodes = nodes[:nnodes]
    edges = [combinations[i] for i in torch.randperm(len(combinations), generator=rng)[:nedges]]

    g = gm.Graph()
    for i in range(nnodes):
        g.add_factor(torch.ones(cardinality[i], names=[nodes[i]]))
    for edge in edges:
        f = torch.rand([cardinality[i] for i in edge], names=[nodes[i] for i in edge])
        g.add_factor(f)
    return g
