import gmtorch as gm
import numpy as np
import torch
from collections import defaultdict
import warnings


def summation(*gs):
    """
    Work in progress...
    """

    result = gm.Graph()
    all_factors = {}
    for g in gs:
        for names in g.factors.keys():
            if names not in all_factors:
                all_factors[names] = []
            all_factors[names].append(g.factors[names].align_to(*names))
    for names in all_factors:
        if len(all_factors[names]) == 1:
            result.add_factor(all_factors[names][0])
            continue
        newnames = list(names) + ['{}_sum'.format(name) for name in names]
        shape = [max([f.shape[i] for f in all_factors[names]]) for i in range(len(names))]
        x = torch.zeros(shape+[1]*len(names))
        x = x.repeat(*[1]*len(names), *[len(gs)]*len(names))
        for i, f in enumerate(all_factors[names]):
            x[[slice(None)]*len(names) + [i]*len(names)] = f.rename(None)
        x.names = newnames
        print(x.shape, x.names)
        result.add_factor(x)
    return result
        # assert 0
        # x = torch.zeros([sum([f.shape[i] for f in all_factors[names]]) for i in range(len(names))])
        # idxs = torch.zeros(len(names)).long()
        # for f in all_factors[names]:
        #     x[[slice(idxs[i], idxs[i]+f.shape[i]) for i in range(len(names))]] += f
        #     idxs += torch.tensor(f.shape)
        # x.names = all_factors[names][0].names
        # print(x)
    # print(all_factors)


def multiplication(*gs, inputs):
    """
    Create a graph representing the product of two or more graphs, such that
        result[inputs] = g1[inputs] * g2[inputs]

    :param gs: a list of `Graph`s, separated by commas
    :param inputs: a list of nodes. Must be included in both graphs
    :return: a new `Graph`
    """

    result = gm.Graph()
    name_count = defaultdict(lambda: 0)
    for g in gs:
        for n in g.nodes:
            name_count[n] += 1
    for n in inputs:
        assert name_count[n] >= 1
    for i, g in enumerate(gs):
        # assert all([n in g.nodes for n in inputs])
        for f in list(g.get_factors()):
            newvariables = list(f.names)
            for n in range(len(newvariables)):
                if newvariables[n] not in inputs:
                    if name_count[newvariables[n]] > 1:  # Repeated nodes get a disambiguating suffix
                        newvariables[n] += '_{}'.format(i+1)
            result.add_factor(f.rename(*newvariables))
    return result


def set_evidence(g, evidence, batch_name='batch'):
    """
    Like `Graph.set_evidence()`, but returning a copy.
    """

    g = g.clone()
    g.set_evidence(evidence=evidence, batch_name=batch_name)
    return g


def squeeze(g):
    """
    Like `Graph.squeeze()`, but returning a copy.
    """

    g = g.clone()
    g.squeeze()
    return g


def division(g1, g2):
    """
    Create a graph representing the quotient of two or more graphs, such that
        result[x] = g1[x] / g2[x]

    :param g1: a `Graph`
    :param g2: a `Graph`
    :return: a `Graph`
    """

    result = g1.clone()
    for f in g2.get_factors():
        if torch.min(torch.abs(f)) == 0:
            warnings.warn('Division by zero (factor {} in denominator graph)'.format(f.names))
        result.add_factor(1/f)
    return result
