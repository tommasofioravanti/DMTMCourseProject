import pandas as pd
import numpy as np
#import statsmodels

from features.exponential_moving_average import exponential_weighted_moving_average
from features.moving_average import moving_average
from features.slope import slope
from features.week_of_the_year import week_of_the_year
from features.season import season
from features.lag_target import lag_target
from features.lag_pos import lag_pos
from features.lag_volume import lag_volume
from features.partial_sales import partial_sales
from features.heavy_light import heavy_light
from features.days_to_christmas import days_to_christmas
from features.tot_price_imp import tot_price_per_wk
from features.price_change import price_change
from features.sales_per_brand_w1 import sales_per_brand_w1
from features.POS_Corr import Corr_Pos
from features.Vol_Corr import Corr
from features.clustering import get_cluster
from features.GaussianTargetEncoding import run_gte_feature

import os
from pathlib import Path


def dfs_gen(df, dates):
    """
    Train-Test generator
    :param df:
    :param val_dates:
    :return:
    """

    df_dates = dates.sort_values()

    df = df.sort_values(['sku','Date']).reset_index(drop=True)
    for d in df_dates:
        yield df[df.Date < d], df[df.Date == d]


def add_all_features(df):
    df = df.sort_values(['Date','sku']).reset_index(drop=True)
    # Features
    df = moving_average(df, 20)
    _, df['increment'] = slope(df)
    df['exp_ma'] = exponential_weighted_moving_average(df, com=0.3)
    # df = lag_target(df, 2)
    # df = lag_target(df, 3)
    # df = lag_target(df, 4)
    # df = lag_target(df, 5)
    # df = lag_target(df, 52)
    df = lag_target(df, 25)
    df = lag_target(df, 50)

    df = lag_pos(df, 1)
    # df = lag_volume(df, 1)
    # df = price_change(df)
    # df = sales_per_brand_w1(df)

    ## Date Features
    # train['year'] = train.Date.dt.strftime('%Y %m %d').str.split(" ").apply(lambda x:x[0]).astype(int)
    df['month'] = df.Date.dt.strftime('%Y %m %d').str.split(" ").apply(lambda x: x[1]).astype(int)
    df['day'] = df.Date.dt.strftime('%Y %m %d').str.split(" ").apply(lambda x: x[2]).astype(int)
    df['year'] = df.Date.dt.strftime('%Y %m %d').str.split(" ").apply(lambda x: x[0]).astype(int)

    df = days_to_christmas(df)
    df = heavy_light(df)
    df = partial_sales(df)

    # Cluster
    cluster = get_cluster()
    df = df.merge(cluster, how='left', on='sku')

    # Gaussian Target Encoding
    abs_path = Path(__file__).absolute().parent
    gte_path = os.path.join(abs_path, "features/gte_features_w8_prp50.csv")
    if os.path.isfile(gte_path):
        gte = pd.read_csv(gte_path)
    else:
        print('Generate Target Encoding Feature')
        run_gte_feature()
        gte = pd.read_csv(gte_path)

    gte.Date = pd.to_datetime(gte.Date)
    df = df.merge(gte, how='left', on=['Date', 'sku', 'target', 'real_target'])

    df = week_of_the_year(df)

    # Season
    df = season(df)
    #Correlation Price-Sales
    #df=conc_corr(df)

    #df=df.dropna() #Use this for Random Forest
    categorical_features = ['cluster', 'heavy_light']

    return df, categorical_features




def get_weights(train, type=0):
    """
    Sample Weights
    :param train:
    :param type: integer that define the generation of the sample weights
    :return:
    """
    if type==0:                 # Si assegna un peso ai samples in base al target del sample e alla media dei target di quello sku
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

    elif type==1:               # Si cerca di pesare di più gli ultimi samples [temporalmente]
        df = train.copy()
        df['position'] = np.ones(df.shape[0], dtype='int')
        df['position'] = df[['Date', 'sku', 'position']].groupby(['sku']).cumsum()
        df = df.sort_values(['sku', 'Date'])
        w = []
        for s in set(df.sku):
            w.append(df[df.sku == s]['position'].values / 100)
        w = [item for x in w for item in x]
        return w
    
    elif type==2:               # Si cerca di pesare di più gli ultimi samples [temporalmente]
        w = []
        for s in train.scope:
            if s==1:
                w.append(1)
            else:
                w.append(5)
        return w

    elif type==3:  # Si pesa di più i samples nella finestra temporale del test

        w = []

        for index, row in train.iterrows():
            # print(row)
            if (not ((str(row.Date) <= '2017-12-14') & (str(row.Date) >= '2017-06-29')) and not (
                    (str(row.Date) <= '2018-12-14') & (str(row.Date) >= '2018-06-29'))):
                w.append(1)
            else:
                w.append(10)
        return w
