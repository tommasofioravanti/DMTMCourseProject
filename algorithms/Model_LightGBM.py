import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

from preprocessing.preprocessing import convert_date, inverse_interpolation
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

    # TODO Add Season

    return df


# The log function essentially de-emphasizes very large values.
# It is more easier for the model to predict correctly if the distribution is not that right-skewed which is
# corrected by modelling log(sales) than sales.
df['target'] = np.log1p(df.target.values)
df['sales w-1'] = np.log1p(df['sales w-1'].values)

df = add_all_features(df)

# Train-Test split
train = df[~df.target.isna()]
# test = df[df.target.isna()]

# Train-Validation split
val_sku = list(set(train[train.scope==1].sku))

## Get the last k % of the last dates
train_dates = train.sort_values('Date').drop_duplicates('Date', keep='first').Date.values
k = int(np.floor(len(train_dates) *  0.20))
val_dates = train_dates[-k:]
mask = train.Date.isin(val_dates)

val = train[mask]
train_ = train[~mask]

# Model
X_train = train_.drop(['target', 'scope', 'Date'], axis=1)
y_train = train_.target

X_val = val.drop(['target', 'scope', 'Date'], axis=1)
y_val = val.target

lgb = LGBMRegressor(metric='mape')

lgb.fit(X_train, y_train, categorical_feature=['sku', 'pack', 'brand'])
predictions = lgb.predict(X_val)

mape = MAPE(y_val, predictions)
print(f'MAPE={mape}')


# Plot Feature Importance
plt.figure(figsize=(8,5))
plt.xticks(rotation=90)
plt.plot(train_.drop(['Date','scope', 'target'], axis=1).columns, lgb.feature_importances_)
plt.show()