import gmtorch as gm
import numpy as np
import torch


def from_tntorch(t, names=None, rank_label=''):
    """
    Cast a tensor network from `tntorch` into a graph.

    See https://github.com/rballester/tntorch

    :param t: a `tntorch.tensor`
    :param names: the names for the physical dimensions of the tensor. If None (default), 'I1', 'I2', etc. will be used
    :param rank_label: optional tag to be attached at the end of each rank node, to avoid collisions. Default is the empty string
    :return: a `Graph`
    """

    if names is None:
        names = ['I{}'.format(n+1) for n in range(t.dim())]
    assert len(names) == t.dim()

    result = gm.Graph()
    rank_counter = 0
    for n in range(t.dim()):
        if t.Us[n] is not None:
            result.add_factor(t.Us[n].rename(names[n], 'S{}{}'.format(n+1, rank_label)))
        if t.cores[n].dim() == 2:
            if t.Us[n] is None:
                result.add_factor(t.cores[n].rename(names[n], 'R{}{}'.format(rank_counter, rank_label)))
            else:
                result.add_factor(t.cores[n].rename('S{}{}'.format(n+1, rank_label), 'R{}{}'.format(rank_counter, rank_label)))
        else:
            if t.Us[n] is None:
                physical = names[n]
            else:
                physical = 'S{}{}'.format(n+1, rank_label)
            if n == 0:
                result.add_factor(t.cores[n][0, ...].rename(physical, 'R{}{}'.format(rank_counter + 1, rank_label)))
            elif n == t.dim()-1:
                result.add_factor(t.cores[n][..., 0].rename('R{}{}'.format(rank_counter, rank_label), physical))
            else:
                result.add_factor(t.cores[n].rename('R{}{}'.format(rank_counter, rank_label), physical, 'R{}{}'.format(rank_counter+1, rank_label)))
            rank_counter += 1
    return result


def from_pgmpy(g):
    """
    Build graph from a `pgmpy` Markov random field network.

    See https://github.com/pgmpy/pgmpy.

    :param g: a `pgmpy.models.MarkovNetwork.MarkovNetwork`
    :return: a `Graph`
    """

    import pgmpy
    result = gm.Graph()
    for factor in g.get_factors():
        newfactor = torch.tensor(factor.values, names=factor.variables)
        result.add_factor(newfactor)
    return result


def to_pgmpy(g):
    """
    Given a graph, build a `pgmpy` Markov random field network.

    See https://github.com/pgmpy/pgmpy.

    :param g: a `Graph`
    :return: a `pgmpy.models.MarkovNetwork.MarkovNetwork`
    """

    import pgmpy
    result = pgmpy.models.MarkovNetwork(g.edges)
    factors = [pgmpy.factors.discrete.DiscreteFactor(
        variables=factor.names,
        cardinality=factor.shape,
        values=factor.numpy()
    ) for factor in g.get_factors()]
    result.add_factors(*factors)
    result.check_model()
    return result


def to_quimb(g):
    """
    Given a graph, build a `quimb` TensorNetwork.

    See https://quimb.readthedocs.io/.

    :param g: a `Graph`
    :return: a `quimb.tensor.tensor_core.TensorNetwork`
    """

    import quimb.tensor as qtn
    tensors = [qtn.Tensor(data=f.detach().numpy(), inds=f.names) for f in g.get_factors()]
    result = tensors[0]
    for t in tensors[1:]:
        result = result & t
    return result