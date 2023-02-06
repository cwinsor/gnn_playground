
# https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html

import os
import pandas as pd
import argparse
import numpy as np
import torch


from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import ShapeNet

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter
import torch_geometric.transforms as T
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from matplotlib import pyplot as plt





def main1():

    x = torch.tensor([1, 2])

    x = torch.tensor(["apple", "baker"], dtype=torch.StringType)

    edge_index = torch.tensor([
        [3, 1, 1, 2],
        [1, 3, 2, 1]], dtype=torch.long)

    x = torch.tensor([
        [-1, "minus one"], [0, "zero"], [1, "one"], [3, "three"]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    print(data)
    data.validate(raise_on_error=True)

# Note that edge_index, i.e. the tensor defining the source and target nodes of all edges,
# is not a list of index tuples. If you want to write your indices this way, you should
# transpose and call contiguous on it before passing them to the data constructor:

def main2():

    edge_index = torch.tensor([
        [8, 1],
        [1, 8],
        [1, 2],
        [2, 1]], dtype=torch.long)
        
    x = torch.tensor([
        [-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index.t().contiguous())
    print(data)
    data.validate(raise_on_error=True)

# Note that it is necessary that the elements in edge_index only hold indices in the range { 0, ..., num_nodes - 1}.
# You can always check that your final Data objects fulfill these requirements by running validate():
def main3():

    edge_index = torch.tensor([
        [0, 1, 1, 3],
        [1, 0, 2, 1]], dtype=torch.long)

    x = torch.tensor([
        [-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    print(data)
    data.validate(raise_on_error=True)



if __name__ == "__main__":
    main1()
    
