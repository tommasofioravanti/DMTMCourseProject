import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

import sys
sys.path.append('.')

from preprocessing.preprocessing import convert_date, inverse_interpolation, train_validation_split
from metrics.MAPE import MAPE
from features.exponential_moving_average import exponential_weighted_moving_average
from features.moving_average import moving_average
from features.slope import slope

train = pd.read_csv("../dataset/original/train.csv")
test = pd.read_csv("../dataset/original/x_test.csv")

df = pd.concat([train, test])
df = convert_date(df)
df = df.sort_values(['sku','Date']).reset_index(drop=True)

# Encoding Categorical Features
le = LabelEncoder()
df.pack = le.fit_transform(df.pack)
df.brand = le.fit_transform(df.brand)

# Impute sales w-1 NaNs in the first week
df = inverse_interpolation(df)


def add_all_features(df):
    # Features
    df = moving_average(df, 20)
    _, df['increment'] = slope(df)
    df['exp_ma'] = exponential_weighted_moving_average(df, com=0.3)

    ## Date Features
    # train['year'] = train.Date.dt.strftime('%Y %m %d').str.split(" ").apply(lambda x:x[0]).astype(int)
    df['month'] = df.Date.dt.strftime('%Y %m %d').str.split(" ").apply(lambda x: x[1]).astype(int)
    df['day'] = df.Date.dt.strftime('%Y %m %d').str.split(" ").apply(lambda x: x[2]).astype(int)

    # Cluster
    cluster = pd.read_csv("../dataset/cluster.csv")
    cluster = cluster.rename(columns={'Label':'cluster', 'Sku':'sku'})
    df = df.merge(cluster, how='left', on='sku')


    # TODO Add Season

    return df


"""The log function essentially de-emphasizes very large values.
It is more easier for the model to predict correctly if the distribution is not that right-skewed which is
corrected by modelling log(sales) than sales."""

real_values = df[['Date', 'sku', 'target']].rename(columns={'target':'real_target'})
df['target'] = np.log1p(df.target.values)
df['sales w-1'] = np.log1p(df['sales w-1'].values)

df = add_all_features(df)

# Train-Test split
train = df[~df.target.isna()]
# test = df[df.target.isna()]

# Train-Validation split
val_sku = list(set(train[train.scope==1].sku))
train, val, val_dates = train_validation_split(train)

# Model
predictions = []
lgb = LGBMRegressor(metric='mape')
drop_cols = ['scope', 'Date']

for d in val_dates:
    print(f'Prediction for week: {str(d.year) + "-" + str(d.month) + "-" + str(d.day) }')

    X_train = train.drop(['target'], axis=1)
    y_train = train.target

    val_week = val[val.Date == d]
    X_val = val_week.drop(['target'], axis=1)
    y_val = val_week.target

    X_train = X_train.drop(drop_cols, axis=1)
    X_val = X_val.drop(drop_cols, axis=1)

    lgb.fit(X_train, y_train, categorical_feature=['sku', 'pack', 'brand'])
    predictions.append(lgb.predict(X_val))

    train = pd.concat([train, val_week])


"""# Model
drop_cols = ['scope', 'Date']
X_train = train.drop(['target'], axis=1)
y_train = train.target

X_val = val.drop(['target'], axis=1)
y_val = val.target

X_train = X_train.drop(drop_cols, axis=1)
X_val = X_val.drop(drop_cols, axis=1)

lgb = LGBMRegressor(metric='mape')

lgb.fit(X_train, y_train, categorical_feature=['sku', 'pack', 'brand'])
predictions = lgb.predict(X_val)
"""

predictions = [p for x in predictions for p in x ]

predictions = np.expm1(predictions)
real_values_val = real_values[real_values.Date.isin(val_dates)]
val = val.merge(real_values_val, how='left', on=['Date', 'sku'])
print(f'Nans in val.real_target {val.real_target.isna().sum()}')

mape = MAPE(val.real_target, predictions)
print(f'\n MAPE={mape}')

val['predictions'] = predictions
print(val[['real_target', 'predictions']])


# Plot Feature Importance
plt.figure(figsize=(8,5))
plt.xticks(rotation=90)
plt.plot(train.drop(drop_cols + ['target'], axis=1).columns, lgb.feature_importances_)
plt.show()

