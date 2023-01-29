'''

Hands-on Graph Neural Networks with PyTorch & PyTorch Geometric
follows:
https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8

with background
https://towardsdatascience.com/a-gentle-introduction-to-graph-neural-network-basics-deepwalk-and-graphsage-db5d540d50b3
https://arxiv.org/ftp/arxiv/papers/1812/1812.08434.pdf

The code that follows is from the "word-embedding" page, not intended to run just EXPLAINED !
See the next test for executible code from that same page ("A Real-World Example — RecSys Challenge 2015")
'''

import os
import pandas as pd
import argparse
import numpy as np
import torch
from torch_geometric.data import Data

def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(
        description="Semisupervised Graph",
        add_help=add_help)

    parser.add_argument("--foo",
        default="bar",
        type=str, help="this is an arg")

    return parser


def main(args):

    print("args:")
    for k, v in dict(vars(args)).items():
        print("{} {}".format(k, v))

    # specify the graph nodes
    # 4 nodes, each having
    #   2 features x0, x1
    #   a class y
    x = torch.tensor([[2,1], [5,6], [3,7], [12,0]], dtype=torch.float)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.float)

    # edge list in COO format:
    #  source_node, target_node
    edge_index = torch.tensor([[0, 1, 2, 0, 3],
                            [1, 0, 1, 3, 2]], dtype=torch.long)

    # create a torch.geometric.data.Data object
    data = Data(x=x, y=y, edge_index=edge_index)

    # propegate(edge_index, node features a.k.a. embeddings)
    # calls
    #   1) MESSAGE - (user-defined taking in ...)
    #   2) UPDATE - takesthe aggregated message, assigns new embedding value for the node
    #
    # example:
    #  1) MESSAGE:
    #   h_kN(v) = AGGREGATE_k({h_uk-1, forall u in neighbors of v})
    #   In other words - look at the neighbors and perform some aggregation on their h values
    #   The example uses max-pooling as AGGREGATE
    #     max( { sigma (Wpool * h_uik + b) for all u in neighbors of v
    #   which can be implemented using a standard single-layer
    #   with sigma a ReLU activation function and Wpool + b as linear layer
    #  2) UPDATE
    #   h_kv = sigma(W_k * CONCAT(h_vk-1, h_N(v)))
    #   In other words - the hidden/embedding for the node is the product of W and the concatenation
    #     of the node's last hidden and the aggregation from the neighbors.
    #   activation is abain implemented as 

class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__(aggr='max') #  "Max" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()
        self.update_lin = torch.nn.Linear(in_channels + out_channels, in_channels, bias=False)
        self.update_act = torch.nn.ReLU()
        
    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        
        
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        # x_j has shape [E, in_channels]

        x_j = self.lin(x_j)
        x_j = self.act(x_j)
        
        return x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]


        new_embedding = torch.cat([aggr_out, x], dim=1)
        
        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)
        
        return new_embedding




if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
