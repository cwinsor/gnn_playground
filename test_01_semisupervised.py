# some exercises derived from the paper
# SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS (Kipf, Welling, 2017)
#
# Equation 1 gives the loss function for a semi-labeled graph.
# First term is the supervised loss
# Second term is the regularization loss where the goal is to
# move nodes that are connected by an edge closer together.
# The equation does this by:
#  b) The L2 distance for each pair of nodes is computed
#  c) Adjacency matrix A filters only nodes that are adjacent (others are zero'd)
#
# As the paper indicates, the regularization loss assumes that nodes that are connected
# have similar values. That, of course, depends on the shape of f(.) which is learned.
#
# What I want to experiment with here is second representation, on the right in
# equation 1. In this is f(X)' DELTA F(X), where DELTA is composed of "D" and "I" matrices.
# D is the degree matrix and I is the identity matrix.



# ALSO SEE (GREAT EXPLANATION)
# https://csustan.csustan.edu/~tom/Clustering/GraphLaplacian-tutorial.pdf

import os
import pandas as pd
import argparse
import numpy as np
import torch


def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(
        description="Semisupervised Graph",
        add_help=add_help)

    parser.add_argument("--foo",
        default="/bar",
        type=str, help="this is an arg")

    return parser


def main(args):

    for k, v in dict(vars(args)).items():
        print("{} {}".format(k, v))

    # number of nodes
    N = 5

    # adjacency matrix
    A = pd.DataFrame(np.zeros((N,N), dtype=np.intc))
    A.loc[0, 1] = 1
    A.loc[0, 2] = 1
    A.loc[1, 3] = 1
    A.loc[1, 4] = 1
    A.loc[2, 4] = 1
    A = A.add(A.T)  # edges are bidirectional
    print("\nA")
    print(A)

    # degree matrix froom adjacency matrix
    def degree_matrix(A):
        out_degree = np.sum(A, axis=0)
        in_degree = np.sum(A, axis=1)
        degree = out_degree + in_degree
        D = np.diag(degree)
        return D
    D = pd.DataFrame(degree_matrix(A))
    print("\nD")
    print(D)

    # vertices
    # f_x is known/given in this example (should be learned)
    # "supervised" is sparsely known (but it is not used in calculating the Lreg part of the loss)
    V = pd.DataFrame(np.random.rand(N,3), columns=('x0', 'x1', 'x2'))
    V['f_x'] = [7., 5., 2., 8., 6.]
    V['supervised'] = [None ,3., 7., None, 1.]
    print("\nV")
    print(V)

    ################################################
    # calculate Lreg via summation

    # Lreg is a NxN square array
    Lreg = pd.DataFrame(np.zeros((N,N)))
    for i in range(N):
        for j in range(N):
            if A.loc[i, j] == 0:
                l2dist = 0
            else:
                t1 = V.loc[i, 'f_x'] 
                t2 = V.loc[j, 'f_x'] 
                l2dist = (t1 - t2) ** 2
            Lreg.loc[i,j] = l2dist

    print("\nLreg")
    print(Lreg)
    print("total Lreg {}".format(Lreg.to_numpy().sum()))

    ################################################
    # now do it again per equation 1 the rightmost expression...

    DELTA = D.subtract(A)
    print("\nDELTA")
    print(DELTA)

    d1 = V['f_x'].T.dot(DELTA)
    d2 = d1.dot(V['f_x'])
    print(d2)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
