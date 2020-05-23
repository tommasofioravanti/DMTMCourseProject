import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append('.')

from features.days_to_christmas import days_to_christmas 
from preprocessing.preprocessing_more import preprocessing_more
from metrics.MAPE import MAPE

# preprocessing
df_train = pd.read_csv('dataset/train.csv')
df_train = preprocessing_more(df_train)

# add features
df_train = days_to_christmas(df_train)

# split data
X_train, X_test, y_train, y_test = train_test_split(
                                        df_train.drop(['target','scope'],axis=1), 
                                        df_train.target, 
                                        test_size=0.2, 
                                        shuffle=False
                                        )

# init model
model = lgb.LGBMRegressor()

# model fitting
model.fit(
    X_train, 
    y_train, 
    categorical_feature=['pack', 'brand','day','year','month','sku']
    )

# performance evaluation
y_pred = model.predict(X_test)
print(f"MAPE: {MAPE(y_test, y_pred):.5f}")
