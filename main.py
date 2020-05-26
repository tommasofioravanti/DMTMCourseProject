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
from features.partial_sales import partial_sales
from features.heavy_light import heavy_light
from features.days_to_christmas import days_to_christmas 

train = pd.read_csv("dataset/original/train.csv")
test = pd.read_csv("dataset/original/x_test.csv")

useTest = True
useScope = True
#   --------------- Preprocessing -----------------

df = pd.concat([train, test])
df = convert_date(df)

if useTest:
    df = df.sort_values('Date')
    df_dates = df[df.target.isna()]
    df_dates = df_dates.drop_duplicates('Date').Date

    # Riga da RIMUOVERE PRIMA DELLA CONSEGNA
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



if useTest:
    gen = dfs_gen(df, df_dates)
else:
    gen = dfs_gen(train, val_dates)

#   --------------- Model -----------------

drop_cols = ['scope', 'Date', 'real_target','pack', 'brand', 'size (GM)']
categorical_f = [x for x in categorical_f if x not in drop_cols]

prediction_df = pd.DataFrame()
pred_cluster = pd.DataFrame()


for df_train, df_test in tqdm(gen):

    model = LightGBM(df_train, df_test, categorical_features=categorical_f, drop_columns=drop_cols, isScope=useScope)
    model_preds = model.run()

    prediction_df = pd.concat([prediction_df, model_preds])

    # ---- Predict by cluster  -----
    drop_cols = drop_cols + ['cluster']
    categorical_f = [x for x in categorical_f if x not in drop_cols]

    cluster_model_1 = LightGBM(df_train[df_train.cluster == 1], df_test[df_test.cluster == 1],
                               categorical_features=categorical_f, drop_columns=drop_cols, name='_c', isScope=useScope)
    cluster_pred_1 = cluster_model_1.run()

    pred_cluster = pd.concat([pred_cluster, cluster_pred_1])

    cluster_model_2 = LightGBM(df_train[df_train.cluster == 2], df_test[df_test.cluster == 2],
                               categorical_features=categorical_f, drop_columns=drop_cols, name='_c', isScope=useScope)
    cluster_pred_2 = cluster_model_2.run()

    pred_cluster = pd.concat([pred_cluster, cluster_pred_2])

    if not useScope:
        cluster_model_3 = LightGBM(df_train[df_train.cluster == 3], df_test[df_test.cluster == 3],
                                   categorical_features=categorical_f, drop_columns=drop_cols, name='_c',
                                   isScope=useScope)
        cluster_pred_3 = cluster_model_3.run()

        pred_cluster = pd.concat([pred_cluster, cluster_pred_3])


prediction_df = prediction_df.merge(pred_cluster, how='left', on=['Date', 'sku', 'target', 'real_target'])
# weighted mean log_predictions
res = []
for l, l_c in tqdm(zip(prediction_df.log_prediction, prediction_df.log_prediction_c)):
    final_pred = (0.4 * l + 0.6 * l_c)
    res.append(final_pred)

prediction_df['final_pred'] = np.expm1(res)


if useTest:
    print(f'Standard MAPE = {MAPE(prediction_df[prediction_df.Date.isin(df_dates[:-1])].real_target,prediction_df[prediction_df.Date.isin(df_dates[:-1])].prediction)}')
    print(f'Cluster MAPE = {MAPE(prediction_df[prediction_df.Date.isin(df_dates[:-1])].real_target,prediction_df[prediction_df.Date.isin(df_dates[:-1])].prediction_c)}')
    print(f'Ensemble MAPE = {MAPE(prediction_df[prediction_df.Date.isin(df_dates[:-1])].real_target,prediction_df[prediction_df.Date.isin(df_dates[:-1])].final_pred)}')


else:
    print(f'Standard MAPE = {MAPE(prediction_df.real_target, prediction_df.prediction)}')
    print(f'Cluster MAPE = {MAPE(prediction_df.real_target, prediction_df.prediction_c)}')
    print(f'Ensemble MAPE = {MAPE(prediction_df.real_target, prediction_df.final_pred)}')

model.plot_feature_importance('Standard')
cluster_model_1.plot_feature_importance('Cluster 1')
cluster_model_2.plot_feature_importance('Cluster 2')

if not useScope:
    cluster_model_3.plot_feature_importance('Cluster 3')
