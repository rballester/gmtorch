import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


def plot(g, inputs=None, show_factors=False, show_cardinality=False, filename=None):
    """
    Visualize a graph using `networkx`.

    :param inputs: optionally, show some input nodes in green. Default is None
    :param show_factors: if True, visualize as a factor graph where potentials are square nodes. Default is False
    :param filename: if None (default), the visualization will be shown. Otherwise, will be saved under this name
    """

    if inputs is None:
        inputs = []

    plt.figure(figsize=(10, 6))

    factornames = []
    if show_factors:
        g2 = nx.Graph()
        g2.add_nodes_from(g)
        for i, f in enumerate(g.get_factors()):
            for n in f.names:
                g2.add_edge('F{}'.format(i+1), n)
                factornames.append('F{}'.format(i+1))
    else:
        g2 = g
        factornames = g.nodes()

    pos = graphviz_layout(g2, prog='neato')  # Positions for the network nodes
    labels = {n: n for n in g2.nodes}
    node_color = ['lightblue' for n in range(len(g.nodes))]
    for i in inputs:
        node_color[list(g.nodes).index(i)] = 'lightgreen'
    nx.draw_networkx_labels(g2, pos=pos, font_size=8, labels=labels)
    nx.draw_networkx_nodes(list(g2.nodes)[:len(g.nodes)], pos=pos, node_color=node_color, node_size=350, linewidths=1, edgecolors='black')
    nx.draw_networkx_nodes(list(g2.nodes)[len(g.nodes):], pos=pos, node_color='black', node_size=150, linewidths=1, edgecolors='black', node_shape='s')
    nx.draw_networkx_edges(g2, pos=pos)
    if show_cardinality:
        if show_factors:
            edge_labels = {(u, v): g.cardinality(u) for u, v in g2.edges if v in factornames}
            nx.draw_networkx_edge_labels(g2, pos, edge_labels=edge_labels, label_pos=1/2, font_color='black', font_size=8)
        else:
            edge_labels = {(u, v): g.cardinality(u) for u, v in g2.edges if v in factornames}
            nx.draw_networkx_edge_labels(g2, pos, edge_labels=edge_labels, label_pos=2/3, font_color='black', font_size=8)
            edge_labels = {(u, v): g.cardinality(v) for u, v in g2.edges if v in factornames}
            nx.draw_networkx_edge_labels(g2, pos, edge_labels=edge_labels, label_pos=1/3, font_color='black', font_size=8)
    plt.axis('off')
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
