# gmtorch - Graphical Modeling in PyTorch

This is a library to do graphical modeling with the advantages of PyTorch, which include easy parallelism, GPU usage, and automatic differentiation. Currently, only Markov random fields are supported. 

`gmtorch` is heavily inspired by the duality between graphical models and tensor networks ([Robeva and Seigal, 2018](https://academic.oup.com/imaiai/article/8/2/273/5041985)), and in fact uses the `opt_einsum` [tensor contraction library](https://optimized-einsum.readthedocs.io/en/stable/) ([Smith and Gray, 2018](https://joss.theoj.org/papers/10.21105/joss.00753)) to marginalize graphical models.

## Installation

You can install *gmtorch* from the source as follows:

```
git clone https://github.com/rballester/gmtorch.git
cd gmtorch
pip install .
```

**Main dependences**:
    - [*NumPy*](https://numpy.org/)
    - [*pgmpy*](https://github.com/pgmpy/pgmpy) (for reading networks and moralizing Bayesian networks)
    - [*PyTorch*](https://pytorch.org/)
    - [*opt_einsum*](https://github.com/dgasmith/opt_einsum)

## Example

`gmtorch` uses named PyTorch tensors (an experimental feature as of now) to encode MRF potentials. Example:

```
import gmtorch as gm
import torch

g = gm.Graph()
g.add_factor(torch.rand(2, 2, names=['A', 'B']))
g.add_factor(torch.rand(2, 3, 4, names=['B', 'C', 'D']))
g.add_factor(torch.rand(3, 5, names=['C', 'E']))
gm.plot(g, show_factors=True, show_cardinality=True)
```

<p align="center"><img src="https://github.com/rballester/gmtorch/blob/main/images/plot_example.jpg" width="600" title="Example graph"></p>

## Tests

We use [*pytest*](https://docs.pytest.org/en/latest/), and the tests depend on [*tntorch*](https://github.com/rballester/tntorch). To run them, do:

```
cd tests/
pytest
```

## Contributing

Pull requests are welcome!

Besides using the [issue tracker](https://github.com/rballester/gmtorch/issues), feel also free to contact me at <rafael.ballester@ie.edu>.
