import torch
import warnings
import gmtorch as gm


def dot(g1, g2):
    """
    Dot product between two graphs, i.e. sum_x(g1[x] g2[x]).

    :param g1: a :class:`Graph`
    :param g2: a :class:`Graph`
    :return: a scalar
    """

    return gm.multiplication(g1, g2, inputs=set(g1.nodes).union(g2.nodes))[[]]


def norm(g):
    """
    L^2 norm of a graph.

    :param g: a :class:`Graph`
    :return: a scalar >= 0
    """

    return torch.sqrt(gm.dot(g, g))


def dist(g1, g2):
    """
    Computes the Euclidean distance between two functions represented as graphs.

    :param g1: a :class:`Graph`
    :param g2: a :class:`Graph`
    :return: a scalar >= 0
    """

    # |a-b|^2 = <a-b, a-b> = <a, a> + <b, b> - 2 <a, b> = |a|^2 + |b|^2 - 2 <a, b>
    return torch.sqrt((gm.dot(g1, g1) + gm.dot(g2, g2) - 2*gm.dot(g1, g2)).clamp(0))


def relative_error(gt, approx):
    """
    Computes the relative error between two graphs.

    :param gt: the groundtruth, a :class:`Graph`
    :param approx: an approximation, a :class:`Graph`
    :return: a scalar >= 0
    """

    dotgt = gm.dot(gt, gt)
    return torch.sqrt((dotgt + gm.dot(approx, approx) - 2 * gm.dot(gt, approx)).clamp(0)) / torch.sqrt(dotgt.clamp(0))


def rmse(gt, approx):
    """
    Computes the root mean squared error (RMSE) error between two graphs.

    :param gt: the groundtruth, a :class:`Graph`
    :param approx: an approximation, a :class:`Graph`
    :return: a scalar >= 0
    """

    return gm.dist(gt, approx) / torch.sqrt(gt.numel())
