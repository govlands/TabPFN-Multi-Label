import openml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabpfn_client import init, TabPFNClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from mlp_att import tabpfn_pred_probas

dataset = openml.datasets.get_dataset(41471)
X, _, _, _ = dataset.get_data(dataset_format='dataframe')
X = X.sample(frac=1).reset_index(drop=True).head(100)

cols = X.columns[-6:]
y = X[cols].map(lambda v: 1 if v else 0)
X = X.drop(columns=cols)

# 数值化、划分
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y.values, test_size=0.2, random_state=42
)

init()
probas = tabpfn_pred_probas(X_train, y_train, X_test)
print(probas)