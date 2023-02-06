
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

    edge_index = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1]], dtype=torch.long)

    x = torch.tensor([
        [-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    print(data)

# Note that edge_index, i.e. the tensor defining the source and target nodes of all edges,
# is not a list of index tuples. If you want to write your indices this way, you should
# transpose and call contiguous on it before passing them to the data constructor:

def main2():

    edge_index = torch.tensor([
        [0, 1],
        [1, 0],
        [1, 2],
        [2, 1]], dtype=torch.long)
        
    x = torch.tensor([
        [-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index.t().contiguous())
    print(data)

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


    # Besides holding a number of node-level, edge-level or graph-level attributes, 
    # Data provides a number of useful utility functions
def main4():

    edge_index = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1]], dtype=torch.long)

    x = torch.tensor([
        [-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)

    print(data.keys)
    print(data['x'])
    for key, item in data:
        print(f'{key} found in data')

    print('edge_attr' in data)
    print(data.num_nodes)
    print(data.num_edges)
    print(data.num_node_features)
    print(data.has_isolated_nodes())
    print(data.has_self_loops())
    print(data.is_directed())

    # Transfer data object to GPU.
    device = torch.device('cuda')
    data = data.to(device)




def print_summary(dataset):
    print(dataset)
    print("number of graphs ", len(dataset))
    print("num_node_features: ", dataset.num_node_features)
    print("num_edge_features: ", dataset.num_edge_features)
    print("num features:      ", dataset.num_features)
    print("num classes:       ", dataset.num_classes)

    data = dataset[0]
    print("data_is_undirected: ", data.is_undirected())
    print(" first graph: ", dataset[0])
    if len(dataset)>1:
        print("second graph: ", dataset[1])



    # Common Benchmark Datasets
    # PyG contains a large number of common benchmark datasets, e.g., all Planetoid datasets (Cora, Citeseer, Pubmed), 
def main5():
    dataset = TUDataset(root=r'/mnt/d/dataset_ENZYMES', name='ENZYMES')
    print_summary(dataset)

# download Cora, the standard benchmark dataset for semi-supervised graph node classification:
def main6():
    dataset = Planetoid(root=r'/mnt/d/dataset_Cora', name='Cora')
    print_summary(dataset)

    # This time, the Data objects holds a label for each node, and additional
    # node-level attributes: train_mask, val_mask and test_mask, where
    data = dataset[0]
    print("is_undirected: ", data.is_undirected())
    print("train_mask.sum(): ", data.train_mask.sum().item())
    print("val_mask.sum(): ", data.val_mask.sum().item())
    print("test_mask.sum(): ", data.test_mask.sum().item())


# Mini-batches
# Neural networks are usually trained in a batch-wise fashion. PyG achieves
# parallelization over a mini-batch ... 
# torch_geometric.loader.DataLoader takes care of this
def main7():
    dataset = TUDataset(root=r'/mnt/d/dataset_ENZYMES', name='ENZYMES', use_node_attr=True)
    print_summary(dataset)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in loader:
        print("batch: ", batch)
        print("num_graphs: ", batch.num_graphs)
        break

# torch_geometric.data.Batch inherits from torch_geometric.data.Data and contains an additional attribute called batch.
# You can use it to, e.g., average node features in the node dimension for each graph individually:
def main8():
    dataset = TUDataset(root=r'/mnt/d/dataset_ENZYMES', name='ENZYMES', use_node_attr=True)
    print_summary(dataset)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    n = 0
    for data in loader:
        print("\ndata: ", data)
        print("num_graphs: ", data.num_graphs)
        print(data.batch)

        x = scatter(
            src=data.x,
            index=data.batch,
            dim=0,
            reduce='mean')
        print(x.shape)
        print(x[0].shape)
  
        # Creating dataset
        a = np.random.randint(100, size =(50))
        
        # Creating plot
        fig = plt.figure(figsize =(10, 7))
        
        plt.hist(a, bins = [0, 10, 20, 30,
                            40, 50, 60, 70,
                            80, 90, 100])
        
        plt.title("Numpy Histogram")
        
        # show plot
        plt.show()


        if n==0:
            break
        n+=1


# Data Transforms
# Transforms are a common way in torchvision to transform images and perform augmentation. 
# PyG comes with its own transforms, which expect a Data object as input and return a new
# transformed Data object. Transforms can be chained together using torch_geometric.transforms.Compose
def main9():
    dataset = ShapeNet(
        root=r'/mnt/d/dataset_ShapeNet',
        categories=['Airplane', 'Chair'])
    print()
    print(dataset)
    print(dataset[0])
    print(dataset[-1])

    # We can convert the point cloud dataset into a graph dataset by generating nearest neighbor
    # graphs from the point clouds via transforms:
    dataset = ShapeNet(
        root=r'/mnt/d/dataset_ShapeNet',
        categories=['Airplane', 'Chair'],
        pre_transform=T.KNNGraph(k=6))
    print()
    print(dataset)
    print(dataset[0])
    print(dataset[-1])

    # In addition, we can use the transform argument to randomly augment a Data object, e.g., 
    # translating each node position by a small number:
    dataset = ShapeNet(
        dataset = ShapeNet(
        root=r'/mnt/d/dataset_ShapeNet',
        categories=['Airplane', 'Chair'],
        pre_transform=T.KNNGraph(k=6)),
        transform=T.RandomJitter(0.01))

# Learning Methods on Graphs
# Use a simple GCN layer and replicate the experiments on the Cora citation dataset. 
# For a high-level explanation on GCN, have a look at its blog post http://tkipf.github.io/graph-convolutional-networks/
def main10():
    dataset = Planetoid(root=r'/mnt/d/dataset_Cora', name='Cora')
    print_summary(dataset)

    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(dataset.num_node_features, 16)
            self.conv2 = GCNConv(16, dataset.num_classes)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index

            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)

            return F.log_softmax(x, dim=1)

    # train the model 200 epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN().to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()  # set the model into 'train' mode
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    # evaluate the model on the test nodes:    
    model.eval()  # set the model into 'eval' mode
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')


# Load the "IMDB-BINARY" dataset from the TUDataset benchmark suite and randomly
# split it into 80%/10%/10% training, validation and test graphs.
def main11():
    dataset = TUDataset(root=r'/mnt/d/dataset_IMDB-BINARY', name='IMDB-BINARY')
    print_summary(dataset)

    train_sz = int(0.8 * len(dataset))
    test_sz = int(0.1 * len(dataset))
    val_sz = len(dataset) - train_sz - test_sz
    print(train_sz, test_sz, val_sz)

    train_dataset = dataset[:train_sz]
    test_dataset = dataset[train_sz: train_sz+test_sz]
    val_dataset = dataset[train_sz+test_sz:]


if __name__ == "__main__":
    main6()
    

