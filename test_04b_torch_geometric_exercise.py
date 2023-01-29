'''

Hands-on Graph Neural Networks with PyTorch & PyTorch Geometric
("A Real-World Example — RecSys Challenge 2015")

follows:
https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8

'''

import os
import pandas as pd
import argparse
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

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

    df = pd.read_csv(r'/mnt/d/dataset_recsys_2015/recsys_2015/yoochoose-clicks.dat', header=None)
    df.columns=['session_id','timestamp','item_id','category']

    buy_df = pd.read_csv(r'/mnt/d/dataset_recsys_2015/recsys_2015/yoochoose-buys.dat', header=None)
    buy_df.columns=['session_id','timestamp','item_id','price','quantity']

    item_encoder = LabelEncoder()
    df['item_id'] = item_encoder.fit_transform(df.item_id)
    print(df.head())
    print(df.info())

    #randomly sample a couple of sessions
    sampled_session_id = np.random.choice(df.session_id.unique(), 20, replace=False)
    print(sampled_session_id)
    criterion = df['session_id'].isin(sampled_session_id)
    df = df.loc[criterion]

    # To determine the ground truth, i.e. whether there is any buy event for a given session,
    # we check if a session_id in yoochoose-clicks.dat presents in yoochoose-buys.dat as well.
    df['label'] = df.session_id.isin(buy_df.session_id)
    print(df)



if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
