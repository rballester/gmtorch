import gmtorch as gm
import numpy as np
import torch


def get_network():
    g = gm.Graph()
    g.add_factor(torch.rand(3, 3, names=['A', 'B']))
    g.add_factor(torch.rand(3, 3, names=['B', 'C']))
    g.add_factor(torch.rand(3, 3, names=['C', 'D']))
    return g


def test_eq1():
    g = get_network()
    g2 = g.clone()
    g2['A'] = 0
    gt = g2[[]]
    assert torch.allclose(gt, g.probability(g.var('A') == 0))


def test_eq2():
    g = get_network()
    gt = 0
    for i in range(3):
        g2 = g.clone()
        g2['A'] = i
        g2['B'] = i
        gt += g2[[]]
    assert torch.allclose(gt, g.probability(g.var('A') == g.var('B')))


def test_lt():
    g = get_network()
    gt = 0
    for i in range(2):
        g2 = g.clone()
        g2['A'] = i
        gt += g2[[]]
    assert torch.allclose(gt, g.probability(g.var('A') < 2))


def test_conditional1():
    g = get_network()
    g2 = g.clone()
    g2['B'] = 0
    denominator = g2[[]]
    g2['A'] = 0
    numerator = g2[[]]
    gt = numerator / denominator
    assert torch.allclose(gt, g.probability(g.var('A') == 0, given=g.var('B') == 0))


def test_conditional2():
    g = get_network()
    denominator = 0
    g2 = g.clone()
    g2['A'] = 0
    numerator = g2[[]]
    denominator += g2[[]]
    g2 = g.clone()
    g2['A'] = 1
    denominator += g2[[]]
    g2 = g.clone()
    g2['A'] = 0
    gt = numerator / denominator
    assert torch.allclose(gt, g.probability(g.var('A') == 0, given=g.var('A') <= 1))
