import gmtorch as gm
import numpy as np
import torch


def test_dot():
    g1 = gm.create_random()
    g2 = gm.create_random()
    gt = g1[...]
    gt = torch.sum(gt*g2[...].align_as(gt))
    ours = gm.dot(g1, g2)
    assert torch.allclose(gt, ours)


def test_norm():
    g = gm.create_random()
    gt = torch.linalg.norm(g[...].rename(None))
    ours = gm.norm(g)
    assert torch.allclose(gt, ours)


def test_relative_error():
    g1 = gm.create_random()
    g2 = gm.create_random()
    full1 = g1[...]
    full2 = g2[...].align_as(full1)
    gt = torch.linalg.norm(full1.rename(None)-full2.rename(None)) / torch.linalg.norm(full1.rename(None))
    ours = gm.relative_error(g1, g2)
    assert torch.allclose(gt, ours)
