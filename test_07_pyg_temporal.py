
# https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html
# https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html#epidemiological-forecasting


import torch

from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.dataset import WikiMathsDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

import torch.nn.functional as F

from torch_geometric_temporal.nn.recurrent import TGCN
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric_temporal.nn.recurrent import AGCRN
from torch_geometric_temporal.nn.recurrent import GConvGRU

from torch_geometric_temporal.nn.recurrent import GConvGRU

from tqdm import tqdm
import json
from six.moves import urllib
import numpy as np


# def summarize_data(data):
#     print(data)
#     print(type(data))
#     print("number of graphs ", len(data))
#     print("num_node_features: ", data.num_node_features)
#     print("num_edge_features: ", data.num_edge_features)
#     # print("num_node_types: ", data.num_node_types)
#     # print("num_edge_types: ", data.num_edge_types)
#     print("num_faces (faces in the mesh): ", data.num_faces)
#     print("num classes: ", data.num_classes)

#     print("num_nodes", data.num_nodes)
#     print("num_edges", data.num_edges)
#     print("size of adjacency matrix", data.size)

#     print("node_attrs()", data.node_attrs())
#     print("edge_attrs()", data.edge_attrs())

#     print("has_isolated_nodes()", data.has_isolated_nodes())
#     print("has_self_loops()", data. has_self_loops())
#     print("is_coalesced()", data.is_coalesced())
#     print("is_cuda()", data.is_cuda())
#     print("is_directed()", data.is_directed())
#     print("is_undidrected()", data.is_undidrected())
#     print("keys", data.keys)

#     print("---")
#     data = data[0]
#     print(type(data))
#     print("data_is_undirected: ", data.is_undirected())
#     print(" first graph: ", data[0])
#     if len(data)>1:
#         print("second graph: ", data[1])


def main_chicken_pox():

    # url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/chickenpox.json"
    # foo = json.loads(urllib.request.urlopen(url).read())
    # print(type(foo))

    # print(type(foo['edges']))
    # print(foo['edges'][0])

    # print(type(foo['node_ids']))
    # print(foo['node_ids']['BEKES'])

    # bar = foo['FX']
    # print(type(bar), len(bar))
    # bar = foo['FX'][0]
    # print(type(bar), len(bar))
    # bar = foo['FX'][0][0]
    # print(type(bar))


    # for k,v in foo.items():
    #     print(k, type(v))

    # print(foo['FX'][0])

    loader = ChickenpoxDatasetLoader()
    dataset = loader.get_dataset()
    print("--- chicken pox dataset---")
    print(type(dataset))
    print_summary(dataset)
    print(len(loader))

    print("edges", loader._edges.shape)
    print("edge_weights", loader._edge_weights.shape)
    print("type(features)", type(loader.features))
    print("features", len(loader.features))
    for n,f in enumerate(loader.features):
        if n==0 or n==1:
            print(type(f))
            print(f.shape)
            print(f)

    print("targets", len(loader.targets))
    print("target[0]", loader.targets[0])


    # print_summary(dataset)
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)
    print("--- train, test ---")
    print(type(train_dataset))
    print(type(test_dataset))

    class RecurrentGCN(torch.nn.Module):
        def __init__(self, node_features):
            super(RecurrentGCN, self).__init__()
            

            # GConvGRU(
            #     in_channels=node_features,
            #     out_channels=32,
            #     K=3,
            #     normalization='sym',
            #     bias=True)

            # self.recurrent = TGCN(
            #     in_channels=node_features,
            #     out_channels=32,
            #     improved=False,
            #     cached=False,
            #     add_self_loops=True)

            self.recurrent = DCRNN(
                in_channels=node_features,
                out_channels=32,
                K=1,
                bias=True)

            # AGCRN(
            #     number_of_nodes=,
            #     in_channels=node_features,
            #     out_channels=32,
            #     K=1,
            #     embedding_dimensions=)


            self.linear = torch.nn.Linear(32, 1)

        def forward(self, x, edge_index, edge_weight):
            h = self.recurrent(x, edge_index, edge_weight)
            h = F.relu(h)
            h = self.linear(h)
            return h

    model = RecurrentGCN(node_features = 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in tqdm(range(200)):
        cost = 0
        for time, snapshot in enumerate(train_dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            cost = cost + torch.mean((y_hat-snapshot.y)**2)
        cost = cost / (time+1)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
    print("training done")

    model.eval()
    cost = 0
    for time, snapshot in enumerate(test_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
    cost = cost / (time+1)
    cost = cost.item()
    print("MSE: {:.4f}".format(cost))



def main_web_traffic():

    loader = WikiMathsDatasetLoader()
    dataset = loader.get_dataset(lags=14)
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.5)

    # investigate a bit...
    print("--- original dataset ---")
    print("type(_dataset): ", type(loader._dataset))
    # print("keys(dataset): ", loader._dataset.keys())
    print("the dictionary keys are:  'edges', 'weights', node_ids', time_periods', and '0..730")
    print("_dataset[time periods]: ", loader._dataset["time_periods"])
    print("_dataset[node_ids]: ", type(loader._dataset["node_ids"]), len(loader._dataset["node_ids"]))
    # print("dataset[node_ids]: ", loader._dataset["node_ids"])
    print("_dataset[node_ids]['Complex number']): ", loader._dataset["node_ids"]['Complex number'])
    # print("_dataset[node_ids].values(): ", loader._dataset["node_ids"].values())
    y = [int(x) for x in loader._dataset["node_ids"].values()]
    print("node_ids hist (5 bins):", np.histogram(y, bins=5))
    print("_dataset[edges]: ", type(loader._dataset["edges"]), len(loader._dataset["edges"]))
    print("_dataset[edge[0]: ", loader._dataset["edges"][0])
    print("_dataset[weights: ", type(loader._dataset["weights"]), len(loader._dataset["weights"]))
    print("_dataset[weights[0]: ", loader._dataset["weights"][0])
    print("_dataset[weights hist (5 bins):", np.histogram(loader._dataset["weights"], bins=5))
    # get histogram of all web-page visit counts
    targets = []
    for time in range(0, loader._dataset["time_periods"]):
        targets.append(np.array(loader._dataset[str(time)]["y"]))
    print("visit counts hist (5 bins):", np.histogram(targets, bins=5))

    print(type(targets))
    print(len(targets))
    print(len(targets[0]))
    print("find sites with max average traffic")
    # a) into 2-d array format        -> count indexed by (timepoint/sitenumber) (stack)
    # b) normalize by timepoint (row) -> normcount indexed by (timepoint/sitenumber)
    # c) reduce using mean across timepoint  -> mean-normalized-count indexed by (sitenumber)
    # --> we have mean normalized count by site number...
    # histogram this
    # d) sort decreasing to get site IDs with highest counts ->  sitenumbers indexed by (smallest-to-largest) 
    # e) convert site ID into site name using the "node_ids" table  -> sitename indexed by (smallest-to-largest)
    a=np.stack(targets)  # should be 731x1068
    print(type(a), a.shape)
    b = a/a.sum(axis=1, keepdims=True)  # should be 731x1068 with each row summing to 1
    print(type(b), b.shape)
    print("check... row sums to 1.0", b[0,:].sum())
    c = np.mean(b, axis=1)  # should be 1068 websites
    print(type(c), c.shape)
    # histogra...
    print("histogram - mean normalized count per site ")
    print(np.histogram(c, bins=5))

    # sort decreasing getting index (website IDs)
    # c = np.array([4,5,6,7,2,8,9])
    d = np.argsort(c)
    d_least_common =  d[0:5]
    d_most_common = d[-5:]
    print("least common ", d_most_common)
    print("most common ", d_least_common)
    # look up website name from index
    # to do this we need inverse lookup (site name from site ID)
    siteid_from_sitename = loader._dataset["node_ids"]
    print("check... Linear programming -> ", siteid_from_sitename["Linear programming"])
    sitename_from_siteid = {v: k for k, v in siteid_from_sitename.items()}
    print("check... 233 -> ", sitename_from_siteid[233])

    # y = [x for x in (1,2,4)]
    # print(y)
    # print(type(sitename_from_siteid))
    # e = [sitename_from_siteid.get(key) for key in [233, 233]]
    # print(e)

    e_most_common = [sitename_from_siteid.get(key) for key in d_most_common]
    e_least_common = [sitename_from_siteid.get(key) for key in d_least_common]

    print("Least common: ", e_least_common)
    print("Most common: ", e_most_common)

    # e = loader._dataset["node_ids"][0]
    # print(e)

    # a number and value as its cube
    dict_created = {0: 0, 1: 1, 2: 8, 3: 27,
                    4: 64, 5: 125, 6: 216}
    # printing type of dictionary created
    print(type(dict_created))
    # converting dictionary to 
    # numpy array 
    res_array = np.array(list(dict_created.items()))
    # printing the converted array
    print(res_array)
    # printing type of converted array
    print(type(res_array))



    for k,v in loader._dataset["node_ids"].items():
        print(k, v)
    print(loader._dataset["node_ids"])

    print(d)
    sorted()
    print()
    print("The *original* Wikipedia Math Essentials Data Set...")
    print("731 timeperiods")
    print("Node list and edge list do not change.")
    print("Attribute lists for node and edge do not change")
    print("Each of the 731 timeperiods has 1068 nodes (web pages) each having a single attrribute that is the visit count")
    print("The edge weights do not change, but they are not just 0/1 - they range between 1 and 16")
    print("what do edge weights represent ???")
    print("The visit activity is highly skewed - 6 pages account for pretty much all the traffic...")


    class RecurrentGCN(torch.nn.Module):
        def __init__(self, node_features, filters):
            super(RecurrentGCN, self).__init__()
            self.recurrent = GConvGRU(node_features, filters, 2)
            self.linear = torch.nn.Linear(filters, 1)

        def forward(self, x, edge_index, edge_weight):
            h = self.recurrent(x, edge_index, edge_weight)
            h = F.relu(h)
            h = self.linear(h)
            return h

    model = RecurrentGCN(node_features=14, filters=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in tqdm(range(50)):
        for time, snapshot in enumerate(train_dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            cost = torch.mean((y_hat-snapshot.y)**2)
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.eval()
    cost = 0
    for time, snapshot in enumerate(test_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
    cost = cost / (time+1)
    cost = cost.item()
    print("MSE: {:.4f}".format(cost))





if __name__ == "__main__":
    # main_chicken_pox()
    main_web_traffic()
    

