import numpy as np

def max_price_per_sku(df_train):
    max_price_per_sku = []

    for s in df_train.sku:
        max_price_per_sku.append(max(df_train[df_train.sku == s]['price']))

    df_train['max_price_per_sku'] = max_price_per_sku
    return df_train


def min_price_per_sku(df_train):
    min_price_per_sku = []

    for s in df_train.sku:
        min_price_per_sku.append(min(df_train[df_train.sku == s]['price']))

    df_train['min_price_per_sku'] = min_price_per_sku
    return df_train


def mean_price_per_sku(df_train):
    mean_price_per_sku = []

    for s in df_train.sku:
        mean_price_per_sku.append(np.mean(df_train[df_train.sku == s]['price']))

    df_train['mean_price_per_sku'] = mean_price_per_sku
    return df_train

if __name__ == '__main__':
    import pandas as pd
    from preprocessing.preprocessing import convert_date

    train = pd.read_csv("../dataset/original/train.csv")
    train = convert_date(train)
    train = max_price_per_sku(train)
    train = min_price_per_sku(train)
    train = mean_price_per_sku(train)

    print(train)
