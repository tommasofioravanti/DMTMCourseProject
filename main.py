import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
from algorithms.Model_LightGBM import LightGBM

import sys
#sys.path.append('.')

from preprocessing.preprocessing import convert_date, inverse_interpolation, train_validation_split
from metrics.MAPE import MAPE
from features.exponential_moving_average import exponential_weighted_moving_average
from features.moving_average import moving_average
from features.slope import slope
from features.week_of_the_year import week_of_the_year
from features.season import season
from features.lag_target import lag_target

train = pd.read_csv("dataset/original/train.csv")
test = pd.read_csv("dataset/original/x_test.csv")

useTest = False
#   --------------- Preprocessing -----------------

df = pd.concat([train, test])
df = convert_date(df)

if useTest:
    df = df.sort_values('Date')
    df_dates = df[df.target.isna()]
    df_dates = df_dates.drop_duplicates('Date').Date
    df.loc[df.target.isna(),'target'] = df[df.target.isna()][['Date', 'sku','sales w-1']].groupby('sku')['sales w-1'].shift(-1).values

df = df.sort_values(['sku','Date']).reset_index(drop=True)

# Encoding Categorical Features
le = LabelEncoder()
df.pack = le.fit_transform(df.pack)
df.brand = le.fit_transform(df.brand)

# Impute sales w-1 NaNs in the first week
df = inverse_interpolation(df)

#   --------------- Features -----------------

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

    # Cluster
    cluster = pd.read_csv("dataset/cluster.csv")
    cluster = cluster.rename(columns={'Label':'cluster', 'Sku':'sku'})
    df = df.merge(cluster, how='left', on='sku')

    df = week_of_the_year(df)

    # Season
    df = season(df)

    categorical_features = ['cluster']
    return df, categorical_features


"""The log function essentially de-emphasizes very large values.
It is more easier for the model to predict correctly if the distribution is not that right-skewed which is
corrected by modelling log(sales) than sales."""

# real_values = df[['Date', 'sku', 'target']].rename(columns={'target':'real_target'})
df['real_target'] = df.target
df['target'] = np.log1p(df.target.values)
df['sales w-1'] = np.log1p(df['sales w-1'].values)

df, categorical_f = add_all_features(df)
categorical_f = ['sku', 'pack', 'brand'] + categorical_f

#   --------------- Split -----------------
df = df.sort_values('Date')

# Train-Test split
train = df[~df.target.isna()]
test = df[df.target.isna()]

# Train-Validation split
_, _, val_dates = train_validation_split(train)



def dfs_gen(df, val_dates=None):
    """
    Train-Test generator
    :param df:
    :param val_dates:
    :return:
    """
    if val_dates is not None:
        df_dates = val_dates.sort_values()
    else:
        df = df.sort_values('Date')
        df_dates = df[df.target.isna()]
        df_dates = df_dates.drop_duplicates('Date').Date

    for d in df_dates:
        yield df[df.Date < d], df[df.Date == d]



if useTest:
    gen = dfs_gen(df, df_dates)
else:
    gen = dfs_gen(train, val_dates)

#   --------------- Model -----------------

drop_cols = ['scope', 'Date', 'real_target']
categorical_f = [x for x in categorical_f if x not in drop_cols]

#train = train.drop(drop_cols, axis=1)
#test = test.drop(drop_cols, axis=1)

prediction_df = pd.DataFrame()

for df_train, df_test in gen:

    model = LightGBM(df_train, df_test, categorical_features=categorical_f ,drop_columns=drop_cols, )
    model_preds = model.run()

    prediction_df = pd.concat([prediction_df, model_preds])

if useTest:
    print(f'MAPE = {MAPE(prediction_df[prediction_df.Date.isin(df_dates[:-1])].real_target,prediction_df[prediction_df.Date.isin(df_dates[:-1])].prediction)}')
else:
    print(f'MAPE = {MAPE(prediction_df.real_target, prediction_df.prediction)}')
model.plot_feature_importance()