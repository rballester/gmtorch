import torch
import operator
import warnings


def log_to_tt(g, order=None, eps=1e-6, rmax=50):
    """
    Cast the logarithm of the MRF as a tensor train using [1]

    [1] Novikov et al. 2014: "Putting MRFs on a Tensor Train" (http://proceedings.mlr.press/v32/novikov14.pdf)

    :param g: a `Graph`
    :param order: list of nodes
    :param eps: default is 1e-6
    :return: a `tntorch.Tensor`
    """

    import tntorch as tn

    ts = []
    for factor in g.get_factors():
        names = factor.names
        if factor.min() < 0:
            raise ValueError('Factor for nodes {} contains negative values'.format(names))
        if factor.min() == 0:
            warnings.warn('Found 0 entry in factor {}. Adding 1e-14 to circumvent it'.format(names))
            factor += 1e-14
        # TODO simplify?
        factor = factor.rename(None)
        factor = factor[(slice(None),)*factor.dim() + (None,)*(g.dim()-factor.dim())]
        factor = factor.rename(*names, *set(g.nodes).difference(set(names)))
        factor = factor.align_to(*order)
        factor = factor.rename(None)
        repeat = [g.cardinality(n) for n in order]
        for n in names:
            repeat[list(order).index(n)] = 1
        t = tn.Tensor(torch.log(factor))
        t = t.repeat(*repeat)
        ts.append(t)
    return tn.reduce(ts, operator.add, eps=eps, rmax=rmax).decompress_tucker_factors()


def graph_hash(g):
    """
    Return a real number that depends on the field represented by this graph (not on the particular choice of factors).

    :param g: a `Graph`
    :return: a scalar
    """

    g = g.clone()
    for n in g.nodes:
        rng = torch.Generator()
        rng.manual_seed(hash(n))
        mult = torch.rand(g.cardinality(n), generator=rng, names=[n])
        mult /= torch.mean(mult)
        g.add_factor(mult)
    return g[[]]
