import os

import pandas as pd


def prepare_data(data_folder):
    x_train = pd.read_csv(os.path.join(data_folder, "x_train.csv"), index_col=0)
    x_test = pd.read_csv(os.path.join(data_folder, "x_test.csv"), index_col=0)
    y_train = pd.read_csv(os.path.join(data_folder, "y_train.csv"))["quality"]
    y_test = pd.read_csv(os.path.join(data_folder, "y_test.csv"))["quality"]

    return x_train, x_test, y_train, y_test
