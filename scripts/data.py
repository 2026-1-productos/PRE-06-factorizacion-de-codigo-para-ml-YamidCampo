import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.25
RANDOM_STATE = 123456
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
DATA_FOLDER = Path("data/winequality-red")


def main():
    df = pd.read_csv(URL, sep=";")
    y = df["quality"].copy()
    X = df.copy()
    X.pop("quality")

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    x_train.to_csv(DATA_FOLDER / "x_train.csv", index=False)
    x_test.to_csv(DATA_FOLDER / "x_test.csv", index=False)
    y_train.to_csv(DATA_FOLDER / "y_train.csv", index=False)
    y_test.to_csv(DATA_FOLDER / "y_test.csv", index=False)

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    main()
