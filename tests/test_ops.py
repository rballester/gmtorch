import gmtorch as gm
import numpy as np
import torch


def test_basic1():
    g = gm.Graph()
    g.add_factor(torch.randn(2, 3, names=['A', 'B']))
    assert g.numel() == 6
    assert g.cardinality('A') == 2
    g.remove_factor(('A', 'B'))
    assert 'A' not in g.nodes
    g.add_factor(torch.randn(4, 5, names=['A', 'B']))
    g.add_factor(torch.randn(5, 6, names=['B', 'C']))
    g.add_factor(torch.randn(4, 6, names=['A', 'C']))
    assert g.cardinality('A') == 4
    assert 'B' in g.neighbors('A')
    assert 'A' in g.neighbors('B')
    g.remove_factor(('A', 'B'))
    assert 'A' in g.nodes


def test_basic2():
    g = gm.Graph()
    g.add_factor(torch.rand(3, 4, names=['A', 'B']))
    g.add_factor(torch.rand(4, 5, names=['B', 'C']))
    g.add_factor(torch.rand(5, 3, names=['C', 'A']))
    assert g.cardinality('A') == 3
    assert g.cardinality('B') == 4
    g.remove_factor(('B', 'A'))
    assert list(g.neighbors('A')) == ['C']
    assert 'A' in g
    g.remove_factor(('A', 'C'))
    assert 'A' not in g
    assert 'B' in g
    assert 'C' in g
    g.remove_factor(('B', 'C'))
    assert len(g.nodes) == 0
    assert len(g.edges) == 0


def test_basic3():
    g = gm.Graph()
    g.add_factor(torch.rand(3, 4, names=['A', 'B']))
    g.add_factor(torch.rand(4, 5, names=['B', 'C']))
    g.add_factor(torch.rand(5, 3, names=['C', 'A']))
    full = g[...]
    assert torch.allclose(g(A=0)[[]], torch.sum(full[0, ...]))


def test_multiply():
    g = gm.Graph()
    g.add_factor(torch.randn(2, 3, names=['A', 'B']))
    g.add_factor(torch.randn(3, 4, names=['B', 'C']))
    g.add_factor(torch.randn(4, 5, names=['C', 'D']))
    g.add_factor(torch.randn(5, 2, names=['D', 'A']))
    gt = g['A', 'B']**2
    ours = gm.multiplication(g, g, inputs=['A', 'B'])['A', 'B']
    assert torch.allclose(gt.rename(None), ours.rename(None))


def test_divide():
    g1 = gm.Graph()
    g1.add_factor(torch.randn(2, 3, names=['A1', 'B1']))
    g1.add_factor(torch.randn(3, 4, names=['B1', 'C1']))
    g1.add_factor(torch.randn(4, 2, names=['C1', 'A1']))
    g2 = gm.Graph()

    # Same factors
    g2.add_factor(torch.randn(2, 3, names=['A1', 'B1']))
    g2.add_factor(torch.randn(3, 4, names=['B1', 'C1']))
    g2.add_factor(torch.randn(4, 2, names=['C1', 'A1']))
    gdiv = gm.division(g1, g2)
    gt = g1[...].rename(None) / g2[...].rename(None)
    ours = gdiv[...].align_to('A1', 'B1', 'C1').rename(None)
    assert torch.allclose(gt, ours)

    # Different factors
    g2 = gm.Graph()
    g2.add_factor(torch.randn(2, 3, names=['A2', 'B2']))
    g2.add_factor(torch.randn(3, 4, names=['B2', 'C2']))
    g2.add_factor(torch.randn(4, 2, names=['C2', 'A2']))
    gdiv = gm.division(g1, g2)
    gt = g1[...].rename(None)[..., None, None, None] / g2[...].rename(None)[None, None, None, ...]
    ours = gdiv[...].align_to('A1', 'B1', 'C1', 'A2', 'B2', 'C2').rename(None)
    assert torch.allclose(gt, ours)


def test_tensor_power():
    import torch
    import tntorch as tn
    t = tn.randn([16]*8, ranks_tt=10)
    names = ['I{}'.format(i + 1) for i in range(t.dim())]
    g = gm.from_tntorch(t, names=names)
    # start = time.time()
    gt = tn.dot(t, t)
    ours = gm.multiplication(g, g, inputs=names)[[]]
    assert torch.allclose(gt, ours)


def test_expand():
    g = gm.Graph()
    f1 = torch.rand(3, 4, names=['A', 'B'])
    f2 = torch.rand(4, 5, names=['B', 'C'])
    g.add_factor(f1)
    g.add_factor(f2)
    gt = torch.einsum('ij,jk->ijk', f1.rename(None), f2.rename(None))
    ours = g['A', 'B', 'C']
    assert torch.allclose(gt, ours.rename(None))


def test_hash():
    g = gm.Graph()
    g.add_factor(torch.rand(3, 4, names=['A', 'B']))
    g.add_factor(torch.rand(4, 5, names=['B', 'C']))
    assert gm.graph_hash(g) == gm.graph_hash(g.clone())
    # assert g == g.clone()


def test_setitem():
    g = gm.Graph()
    f1 = torch.rand(3, 4, names=['A', 'B'])
    f2 = torch.rand(4, 5, names=['B', 'C'])
    g.add_factor(f1)
    g.add_factor(f2)
    g['A', 'B'] = 1
    gt = torch.einsum('ij,jk->', f1[1:2, 1:2].rename(None), f2[1:2, :].rename(None))
    ours = g[[]]
    assert torch.allclose(gt, ours)


def test_arithmetics():
    g = gm.create_random()
    z = g[[]]
    g2 = g*2
    assert torch.allclose(g2[[]], z*2)
    g2 = g/2
    assert torch.allclose(g2[[]], z/2)
    g2 = 1/g
    assert torch.allclose(g2[[]], torch.sum(1/g[...].rename(None)))
    g2 = g**2
    assert torch.allclose(g2[[]], torch.sum(g[...].rename(None)**2))


def test_evidence():
    g = gm.create_random()
    P = 15
    evidence = {node: torch.randint(0, g.cardinality(node), [P]) for node in list(g.nodes)[-5:]}
    g2 = gm.set_evidence(g, evidence)
    # assert all(g2.cardinality(node) == 1 for node in list(g.nodes)[-5:])
    assert all(node not in g2.nodes for node in list(g.nodes)[-5:])
    assert g2.cardinality('batch') == P
    gt = gm.set_evidence(g, {node: evidence[node][0].item() for node in evidence})[[]]
    ours = g2(batch=0)[[]]
    assert torch.allclose(gt, ours)


# g = gm.Graph()
# g.add_factor(torch.rand(3, 4, names=['A', 'B']))
# g.add_factor(torch.rand(4, 5, names=['B', 'C']))
# full = g[...].align_to('A', 'B', 'C')
# print(full[0, 1, :])
# g.set_evidence({'A': [0]})
# g.set_evidence({'B': [1]})
# # g.set_evidence({'C': [0]})
# print(g.cardinality())
# print(g[...])