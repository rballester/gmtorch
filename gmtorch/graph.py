import gmtorch as gm
import numpy as np
import torch
import time
import copy
import networkx as nx
import itertools
import math
import re
import opt_einsum as oe


class Graph(nx.Graph):
    """
    Class for Markov Random Fields (MRFs).

    Note that all nodes and edges must belong to a factor in the graph (no orphan
    nodes/edges allowed).
    """

    def __init__(self, *args, **kwargs):
        self.factors = {}  # Dict: tuple of sorted nodes -> factor (i.e. named PyTorch tensor)
        self.node2factors = {}  # Dict: node -> set of sorted node tuples
        self.edge2factors = {}  # Dict: sorted(u, v) -> set of sorted node tuples
        super().__init__(*args, **kwargs)

    def clone(self):
        """
        See PyTorch's `clone()'
        :return: a `Graph`
        """

        ret = Graph()
        ret.add_nodes_from(self.nodes)
        ret.add_edges_from(self.edges)
        ret.factors = {nodes: self.factors[nodes].clone() for nodes in self.factors}
        ret.node2factors = copy.deepcopy(self.node2factors)
        ret.edge2factors = copy.deepcopy(self.edge2factors)
        return ret

    def detach(self):
        """
        See PyTorch's `detach()'
        :return: a `Graph`
        """

        ret = Graph()
        ret.add_nodes_from(self.nodes)
        ret.add_edges_from(self.edges)
        ret.factors = {nodes: self.factors[nodes].detach() for nodes in self.factors}
        ret.node2factors = copy.deepcopy(self.node2factors)
        ret.edge2factors = copy.deepcopy(self.edge2factors)
        return ret

    def zero_grad(self):
        """
        Set to zero all factor gradients in this `Graph`.
        """

        for nodes in self.factors:
            self.factors[nodes].grad.data = None

    def remove_factor(self, nodes):
        """
        Remove a factor in-place. Nodes and edges that do not belong to other factors will be deleted.

        :param nodes: a tuple of nodes
        """

        nodes = tuple(sorted(nodes))
        if nodes not in self.factors:
            raise ValueError('No factor exists over nodes {}'.format(nodes))

        # Remove from self.factors
        self.factors.pop(nodes)

        # Remove from self.edge2factors
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                edge = tuple(sorted((nodes[i], nodes[j])))
                self.edge2factors[edge].remove(nodes)
                if len(self.edge2factors[edge]) == 0:
                    self.edge2factors.pop(edge)
                    self.remove_edge(*edge)

        # Remove from self.node2factors
        for i in range(len(nodes)):
            self.node2factors[nodes[i]].remove(nodes)
            if len(self.node2factors[nodes[i]]) == 0:
                self.node2factors.pop(nodes[i])
                self.remove_node(nodes[i])

    def add_factor(self, factor):
        """
        Add a new factor in-place. New nodes and edges will be added, if not previously present.

        :param factor: a named `torch.tensor`
        """

        # Check that all factor dimensions match existing cardinalities
        for i in range(factor.dim()):
            if factor.names[i] in self.nodes and factor.shape[i] != self.cardinality(factor.names[i]):
                raise ValueError("Tried to add factor whose cardinality for node {} is {}, but expected {}".format(factor.names[i], factor.shape[i], self.cardinality(factor.names[i])))

        nodes = tuple(sorted(factor.names))

        if nodes in self.factors:  # Duplicate factors are not allowed. The new factor is absorbed
            self.factors[nodes] = self.factors[nodes]*factor.align_as(self.factors[nodes])
            return

        # Add to self.factors
        self.factors[nodes] = factor

        # Add to self.node2factors
        for name in nodes:
            self.add_node(name)
            if name not in self.node2factors:
                self.node2factors[name] = set()
            self.node2factors[name].add(nodes)

        # Add to edge2factors
        for i in range(factor.dim()):
            for j in range(i + 1, factor.dim()):
                self.add_edge(factor.names[i], factor.names[j])
                # self.add_edge(factor.names[j], factor.names[i])
                edge = tuple(sorted((factor.names[i], factor.names[j])))
                if edge not in self.edge2factors:
                    self.edge2factors[edge] = set()
                self.edge2factors[edge].add(nodes)

    def get_factor(self, nodes):
        """
        Return the factor corresponding to a tuple of nodes.

        :param nodes: an iterable
        :return: a tensor
        """

        return self.factors[tuple(sorted(nodes))]

    def get_factors(self, node=None):
        """
        Get all graph factors (or only those that contain a given node).

        :param node: an optional node. Default is None
        :return: a list of factors
        """
        if node is None:
            return list(self.factors.values())
        else:
            assert node in self.nodes
            return list(self.node2factors(node))

    def cardinality(self, node=None):
        """
        Return the cardinality of a certain node.

        :param node: a node of the graph
        :return: a positive integer
        """

        if node is None:
            return {n: self.cardinality(n) for n in self.nodes}
        if node not in self.nodes:
            raise ValueError('Node {} not present in graph'.format(node))
        factor = self.factors[next(iter(self.node2factors[node]))]
        return factor.shape[factor.names.index(node)]

    def dim(self):
        """
        Number of nodes of this graph.

        :return: an integer >= 0
        """

        return len(self.nodes)

    def numel(self):
        """
        Total number of possible node configurations in this graph.

        Example: if g has shape {'A': 3, 'B': 5}, then g.numel() == 15.

        :return: a float64
        """

        return torch.round(torch.prod(torch.tensor([self.cardinality(n) for n in self.nodes]).double()))

    def numcoef(self):
        """
        Total number of elements contained by this graph's factors.

        :return: an integer >= 0
        """

        return sum([f.numel() for f in self.get_factors()])

    def __mul__(self, other):

        if isinstance(other, Graph):
            raise ValueError('For multiplication of two graphs, see `gm.multiplication()`.')

        result = self.clone()
        result.add_factor(torch.tensor(other, names=[]))
        return result

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):

        if isinstance(other, Graph):
            raise ValueError('For division of two graphs, see `gm.division()`.')

        result = self.clone()
        result.add_factor(torch.tensor(1/other, names=[]))
        return result

    def __rtruediv__(self, other):
        result = gm.Graph()
        for f in self.get_factors():
            result.add_factor(other/f)
        return result

    def __pow__(self, other):
        result = gm.Graph()
        for f in self.get_factors():
            result.add_factor(f**other)
        return result

    def squeeze(self):
        """
        Remove singleton dimensions, in-place.
        """

        to_remove = [n for n in self.nodes if self.cardinality(n) == 1]
        self.eliminate(to_remove=to_remove)

    def to(self, target):
        """
        See PyTorch's `to()`.

        :param target: type or device
        :return: a copy with the given target
        """

        result = self.clone()
        for nodes in self.factors:
            result.factors[nodes] = result.factors[nodes].to(target)
        return result

    def __call__(self, **evidence):
        """
        Convenience method to set evidence on a copy graph.

        Example: if g has shape {'A': 3, 'B': 5, 'C': 6}, then g(A=0) has shape {'A': 1, 'B': 1, 'C': 6}.
        """

        return gm.set_evidence(self, evidence)

    def __getitem__(self, item):
        """
        Expand the product of a list of nodes (marginalize the rest).

        Example: if g has shape {'A': 3, 'B': 5}, then g['A'] is a tensor of shape 3.

        :param item: a list of nodes
        :return: a tensor; its names will match `item`
        """

        if isinstance(item, str):
            item = [item]
        if item == Ellipsis:
            item = self.nodes
        nodes = list(self.nodes)
        args = []
        for f in self.get_factors():
            args.append(f.rename(None))
            args.append([nodes.index(n) for n in f.names])
        args.append([nodes.index(n) for n in item])
        return oe.contract(*args, optimize='auto').rename(*item)

    def _contraction_order(self):
        args = []
        for f in self.get_factors():
            args.append(f.rename(None))
            args.append([list(self.nodes).index(n) for n in f.names])
        args.append([])
        path, _ = oe.contract_path(*args)
        print(path, _)
        nodes = list(self.nodes)
        result = []
        for p in path:
            result.append(nodes[p[0]])
            nodes.remove(nodes[p[0]])
        return result + nodes

    def __setitem__(self, key, value):
        """
        Convenience method to set evidence in place. Examples:

        g['A'] = 0
        g['A', 'B'] = 0
        g['A', 'B'] = 0, 1
        """

        if isinstance(key, str) or not hasattr(key, '__len__'):
            key = [key]
        if isinstance(value, str) or not hasattr(value, '__len__'):
            value = [value]*len(key)
        assert len(key) == len(value)
        self.set_evidence({k: v for k, v in zip(key, value)})

    def set_evidence(self, evidence, mode='slice', batch_name='batch'):
        """
        Evaluate one or many graph nodes on given values. If the values are vectors, a new node representing the batch instances will be created.

        Example: suppose the graph has shape {'A': 3, 'B': 4, 'C': 5} and we call this function with evidence {'A': [0, 1], 'B': [1, 2]}, mode='slice', and batch name 'batch'. Then, the graph will have shape {'A': 1, 'B': 1, 'C': 5, 'batch': 2}.

        :param evidence: a dictionary {node: value} (normal mode) or {node: vector of B values} (batched mode)
        :param mode:
            - If 'slice' (default), factors affected by the evidence will be sliced (will become smaller). This makes inference faster later on.
            - If 'mask', the factors will retain their size but will be zeroed-out where relevant. This is good when we want to compute gradients later on.
        :param batch_name: in batch mode, the name of the new node. If a node with that name already exists, a number will be appended (e.g. 'batch' -> 'batch_2'). Default is 'batch'
        :return: a :class:`Graph`
        """

        batched = False
        for ev in evidence.keys():
            if hasattr(evidence[ev], '__len__'):
                batched = True
            else:
                assert not batched  # Mixing batch and non-batch evidence is not allowed
            if ev not in self.nodes:
                raise ValueError('Node {} not present in graph'.format(ev))
            if max(np.atleast_1d(evidence[ev])) >= self.cardinality(ev):
                raise ValueError(
                    'Cannot set node {} to value {} (cardinality is {})'.format(ev, evidence[ev], self.cardinality(ev)))
        while batch_name in self.nodes:  # Pick new name if already existing
            match = re.search('(.*)_(\d+)$', batch_name)
            if match:
                batch_name = '{}_{}'.format(match.group(1), int(match.group(2)) + 1)
            else:
                batch_name += '_2'

        for factor in self.get_factors():
            intersection = set(evidence).intersection(set(factor.names))
            if len(intersection) == 0:  # Factor not affected by the evidence
                continue
            self.remove_factor(factor.names)
            idx = [slice(None) for n in range(factor.dim())]
            names = list(factor.names)
            for ev in intersection:
                idx[factor.names.index(ev)] = evidence[ev]
                names.remove(ev)
            if mode == 'slice':
                if batched:
                    positions = np.sort([factor.names.index(ev) for ev in intersection])
                    if len(positions) > 1 and np.max(np.diff(positions)) > 1:
                        # Non-contiguous fancy indices make PyTorch add the new dimension in the beginning
                        names.insert(0, batch_name)
                    else:
                        names.insert(min(positions), batch_name)
                factor = factor.rename(None)[tuple(idx)]
                factor = factor.rename(*names)
            elif mode == 'mask':
                if batched:
                    raise NotImplementedError
                mask = torch.zeros_like(factor)
                mask[tuple(idx)] = 1
                factor.data *= mask
            else:
                raise ValueError('`mode` should be either slice or mask')
            self.add_factor(factor)

    def __str__(self):
        s = ''
        s += 'Graph with {} nodes, {} edges, {} factors'.format(len(self.nodes), len(self.edges), len(self.factors))
        s += '\n'
        s += 'Largest factors:'
        sizes = {nodes: self.factors[nodes].numel() for nodes in self.factors}
        for i in range(min(20, len(sizes))):
            s += '\n'
            nodes = max(sizes, key=sizes.get)
            s += '\t{} has size '.format(self.factors[nodes].names)
            s += ' x '.join(['{}'.format(sh) for sh in self.factors[nodes].shape])
            s += ' = {}'.format(self.factors[nodes].numel())
            sizes.pop(nodes)
        if len(self.factors) > 20:
            s += '\n'
            s += '\t...'
        s += '\n'
        s += 'Total: {} coefficients'.format(self.numcoef())
        return s

    def requires_grad_(self):
        for f in self.get_factors():
            f.requires_grad_()

    def var(self, node):
        q = Query()
        q.node = node
        q.add_factor(torch.ones(self.cardinality(node), names=[node]))
        return q

    def probability(self, expr, given=None):
        if given is None:
            return gm.multiplication(self, expr, inputs=set(self.nodes).union(expr.nodes))[[]]
        numerator = gm.multiplication(self, expr, given, inputs=set(self.nodes).union(expr.nodes, given.nodes))[[]]
        denominator = self.probability(expr=given, given=None)
        return numerator / denominator


class Query(Graph):

    def _function_graph(self, function, other):
        result = self.clone()
        if isinstance(other, Query):
            for f in other.get_factors():
                result.add_factor(f)
            f = torch.zeros(self.cardinality(self.node), other.cardinality(other.node))
            idx = np.unravel_index(np.arange(f.numel()), f.shape)
            where = np.where(getattr(idx[0], function)(idx[1]))[0]
            f[idx[0][where], idx[1][where]] = 1
            f = f.rename(self.node, other.node)
        else:
            f = torch.zeros(self.cardinality(self.node))
            idx = np.unravel_index(np.arange(f.numel()), f.shape)
            where = np.where(getattr(idx[0], function)(other))[0]
            f[idx[0][where]] = 1
            f = f.rename(self.node)
        result.add_factor(f)
        return result

    def __eq__(self, other):
        return self._function_graph('__eq__', other)

    def __le__(self, other):
        return self._function_graph('__le__', other)

    def __lt__(self, other):
        return self._function_graph('__lt__', other)

    def __ge__(self, other):
        return self._function_graph('__ge__', other)

    def __gt__(self, other):
        return self._function_graph('__gt__', other)

    def __ne__(self, other):
        return self._function_graph('__ne__', other)

