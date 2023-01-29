
import os
import pandas as pd
import argparse
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


df = pd.DataFrame({'Animal': ['Falcon', 'Falcon',
                              'Parrot', 'Parrot'],
                   'MaxSpeed': [380., 370., 24., 26.],
                   'Color': ['green', 'red', 'blue', 'green']}
                   )
print()
print(df)
grouped = df.groupby('Animal')
for key, item in grouped:
    print()
    print(key)
    print(type(item))
    print(item)
    # print(grouped.get_group(key))
    item = item.reset_index(drop=True)
    print(item)

