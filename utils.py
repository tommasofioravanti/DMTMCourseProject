import pandas as pd
import numpy as np

from features.exponential_moving_average import exponential_weighted_moving_average
from features.moving_average import moving_average
from features.slope import slope
from features.week_of_the_year import week_of_the_year
from features.season import season
from features.lag_target import lag_target
from features.partial_sales import partial_sales
from features.heavy_light import heavy_light
from features.days_to_christmas import days_to_christmas


def dfs_gen(df, dates=None):
    """
    Train-Test generator
    :param df:
    :param val_dates:
    :return:
    """
    if dates is not None:
        df_dates = dates.sort_values()
    else:
        df = df.sort_values('Date')
        df_dates = df[df.target.isna()]
        df_dates = df_dates.drop_duplicates('Date').Date

    for d in df_dates:
        yield df[df.Date < d], df[df.Date == d]


def add_all_features(df):
    # Features
    df = moving_average(df, 20)
    _, df['increment'] = slope(df)
    df['exp_ma'] = exponential_weighted_moving_average(df, com=0.3)
    df = lag_target(df, 25)
    df = lag_target(df, 50)

    ## Date Features
    # train['year'] = train.Date.dt.strftime('%Y %m %d').str.split(" ").apply(lambda x:x[0]).astype(int)
    df['month'] = df.Date.dt.strftime('%Y %m %d').str.split(" ").apply(lambda x: x[1]).astype(int)
    df['day'] = df.Date.dt.strftime('%Y %m %d').str.split(" ").apply(lambda x: x[2]).astype(int)
    df['year'] = df.Date.dt.strftime('%Y %m %d').str.split(" ").apply(lambda x: x[0]).astype(int)

    df = days_to_christmas(df)
    df = heavy_light(df)
    df = partial_sales(df)

    # Cluster
    cluster = pd.read_csv("dataset/cluster.csv")
    cluster = cluster.rename(columns={'Label':'cluster', 'Sku':'sku'})
    df = df.merge(cluster, how='left', on='sku')

    df = week_of_the_year(df)

    # Season
    df = season(df)

    categorical_features = ['cluster']
    return df, categorical_features




def get_weights(train):
    weights = []
    sum_dict = {}
    for s, t in zip(train.sku,  train.target):
        if s in sum_dict:
            target_sum = sum_dict[s]
        else:
            target_arr = train[train.sku == s].target.values
            ones = np.ones(target_arr.shape[0])
            target_arr = np.divide(ones, target_arr)
            target_sum = np.sum(target_arr)
            sum_dict[s] = target_sum
        weights.append((1/t)/target_sum)
    return weights

