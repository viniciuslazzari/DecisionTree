import pandas as pd
import numpy as np

from src.node import Node

def main():
    df = pd.read_csv("data/heart.csv")

    h = Node(None, 0)

    train = df.sample(frac = 0.7, random_state = 200)

    test = df.drop(train.index).reset_index(drop = True)
    y_test = test['output']
    x_test = test.loc[:, test.columns != 'output']

    train = train.reset_index(drop = True)
    y_train = train['output']
    x_train = train.loc[:, train.columns != 'output']

    h.create_tree(x_train, y_train, 10)

    true = 0
    false = 0
    for i in range(1, len(x_test)):
        if (h.make_prediction(x_test.iloc[[i]]) == y_test.iloc[[i]]).bool():
            true += 1
        else:
            false += 1

    print(true, false)

if __name__ == "__main__":
    main()