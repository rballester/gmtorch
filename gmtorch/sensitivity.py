import torch
import warnings
import gmtorch as gm


def yodo(g, probability, given=None):
    """
    Implementation of [1]. For every parameter of a graph and some probability of interest (for example, that a variable takes a certain value, optionally conditioned on some other variable), finds its derivative w.r.t. each of its parameters. Note that, by taking the absolute value of that derivative, one obtains the *sensitiviy value* [2].

    Note: this function assumes proportional covariation [3].

    [1] R. Ballester-Ripoll, M. Leonelli: "You Only Derive Once (YODO): Automatic Differentiation for Efficient Sensitivity Analysis in Bayesian Networks". Machine Learning Research, 2022.
    
    [2] L. van der Gaag, S. Renooij, V. Couplé: "Sensitivity analysis of probabilistic networks". In book: Advances in probabilistic graphical models, 2007.

    [3] K. Laskey: "Sensitivity analysis for probability assessments in Bayesian networks". IEEE Transactions on Systems, Man, and Cybernetics, 1995.

    Example:
    >>> g = pgmpy.readwrite.BIFReader("networks/alarm.bif").get_model()
    >>> g = gm.from_pgmpy(g.to_markov_model())
    >>> yodo(g, probability={'CVP': 2}, given={'HISTORY': 1})

    :param g: a `gmtorch.Graph`
    :param probability: a dictionary with one key-value pair: {variable: value}
    :param given: [optional] a dictionary {variable: value} for all conditional evidence 
    :return: a Graph with one factor per original factor; their elements are the original elements' derivatives
    """

    g.requires_grad_()

    numerator = g.detach().clone()
    numerator.requires_grad_()
    denominator = g.detach().clone()
    denominator.requires_grad_()

    if given is None:
        # Marginal probability case: the function of interest if P(Y_O = y_O)
        numerator.set_evidence(probability, mode='mask')
    else:
        # Conditional probability case: the function of interest if P(Y_O = y_O | y_E = y_E)
        numerator.set_evidence({**probability, **given}, mode='mask')
        denominator.set_evidence(given, mode='mask')

    out_numerator = numerator[[]]
    out_numerator.backward()
    out_denominator = denominator[[]]
    out_denominator.backward()

    result = gm.Graph()
    for nodes in numerator.factors.keys():
        f = g.factors[nodes]

        # Find numerator coefficients c_1 and c_2
        grad1 = numerator.factors[nodes].grad
        num = -torch.sum(grad1*numerator.factors[nodes], dim=0, keepdim=True) + grad1*numerator.factors[nodes]
        denom = 1-numerator.factors[nodes]
        grad1 = num / denom + grad1
        c1 = grad1
        c2 = out_numerator.item() - f*c1

        # Find denominator coefficients c_3 and c_4
        grad2 = denominator.factors[nodes].grad
        num = -torch.sum(grad2*denominator.factors[nodes], dim=0, keepdim=True) + grad2*denominator.factors[nodes]
        denom = 1-denominator.factors[nodes]
        grad2 = num / denom + grad2
        c3 = grad2
        c4 = out_denominator.item() - f*c3

        # Compute f'(\theta_i)
        derivative = (c1*c4 - c2*c3) / (f*c3 + c4)**2

        # Add factor containing the derivatives
        result.add_factor(derivative.detach().rename(*f.names))
        
    return result