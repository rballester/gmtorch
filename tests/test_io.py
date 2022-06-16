import gmtorch as gm
import numpy as np
import torch
import tntorch as tn


def test_from_tntorch():
    import torch
    t = tn.rand([10, 10, 10, 10, 10], ranks_tt=5, ranks_tucker=[None, 3, None, 3, 3])
    t.cores[2] = t.cores[2][0, ...]  # Makes it CP along this mode
    g = gm.from_tntorch(t)
    gt = tn.sum(t)
    assert torch.allclose(gt, g[[]])


def test_pgmpy():
    import pgmpy.readwrite
    g = pgmpy.readwrite.BIFReader("/home/rballester/Dropbox/Aplicaciones/Overleaf/graphsobol/Networks/new_networks/dependent_indep.biz").get_model().to_markov_model()
    g2 = gm.from_pgmpy(g).to(torch.float32)
    g3 = gm.to_pgmpy(g2)
    g4 = gm.from_pgmpy(g3).to(torch.float32)
    assert gm.graph_hash(g2) == gm.graph_hash(g4)


def test_gpu():
    if not torch.cuda.is_available():
        return
    g = gm.Graph()
    g.add_factor(2 * torch.rand(3, 3, names=['A', 'B']))
    inputs = ['A']
    cpu = gm.multiplication(g, g, inputs=inputs)[[]]
    g = g.to('cuda')
    cuda = gm.multiplication(g, g, inputs=inputs)[[]]
    assert torch.allclose(cpu, cuda.to('cpu'))
