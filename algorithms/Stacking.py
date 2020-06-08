import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
sys.path.append('../')
from preprocessing.preprocessing import preprocessing

path_train = '../dataset/prediction/val/'
path_test =  '../dataset/prediction/test/'

# Load predictions
lgb_std_train = pd.read_csv(path_train + "lgb_std_val.csv")
lgb_std_test = pd.read_csv(path_test + "lgb_std_test.csv")

lgb_cls_train = pd.read_csv(path_train + "lgb_cls_val.csv")
lgb_cls_test = pd.read_csv(path_test + "lgb_cls_test.csv")

cat_std_train = pd.read_csv(path_train + "cat_std_val.csv")
cat_std_test = pd.read_csv(path_test + "cat_std_test.csv")

cat_cls_train = pd.read_csv(path_train + "cat_cls_val.csv")
cat_cls_test = pd.read_csv(path_test + "cat_cls_test.csv")

#xgb_train = pd.read_csv(path_train + "xgb_incremental_val.csv")
#xgb_test = pd.read_csv(path_test + "xgb_incremental_test.csv")

lin_reg_train = pd.read_csv(path_train + "linear_reg_val.csv")
lin_reg_test = pd.read_csv(path_test + "linear_reg_test.csv")

train = pd.read_csv("../dataset/original/train.csv")
test = pd.read_csv("../dataset/original/x_test.csv")
df = preprocessing(train, test, useTest=False)

# Train
prediction_train = lgb_cls_train.merge(lgb_std_train, how='left', on=['Date', 'sku', 'target', 'real_target'])
prediction_train = prediction_train.merge(cat_std_train, how='left', on=['Date', 'sku', 'target', 'real_target'])
prediction_train = prediction_train.merge(cat_cls_train, how='left', on=['Date', 'sku', 'target', 'real_target'])
#prediction_train = prediction_train.merge(xgb_train, how='left', on=['Date', 'sku', 'target', 'real_target'])
prediction_train = prediction_train.merge(lin_reg_train, how='left', on=['Date', 'sku', 'target', 'real_target'])

prediction_train.Date = pd.to_datetime(prediction_train.Date)
prediction_train = prediction_train.merge(df[['Date', 'sku', 'sales w-1']], how='left', on=['Date', 'sku'])
#prediction_train = prediction_train.merge(gte, how='left', on=['Date', 'sku', 'target', 'real_target'])


# Test
prediction_test = lgb_cls_test.merge(lgb_std_test, how='left', on=['Date', 'sku', 'target', 'real_target'])
prediction_test = prediction_test.merge(cat_std_test, how='left', on=['Date', 'sku', 'target', 'real_target'])
prediction_test = prediction_test.merge(cat_cls_test, how='left', on=['Date', 'sku', 'target', 'real_target'])
#prediction_test = prediction_test.merge(xgb_test, how='left', on=['Date', 'sku', 'target', 'real_target'])
prediction_test = prediction_test.merge(lin_reg_test.drop(['target', 'real_target'], axis=1), how='left', on=['Date', 'sku'])

prediction_test.Date = pd.to_datetime(prediction_test.Date)
prediction_test = prediction_test.merge(df[['Date', 'sku', 'sales w-1']], how='left')
#prediction_test = prediction_test.merge(gte, how='left', on=['Date', 'sku', 'target', 'real_target'])


# In questo giorno il MAPE è particolarmente alto, outlier --> si droppa
mask = (prediction_train.Date=='2017-01-07')
prediction_train = prediction_train.drop(prediction_train[mask].index)


cols = ['log_prediction_lgb_cls',
        'log_prediction_lgb_std',
        'log_prediction_cat_std',
        'log_prediction_cat_cls',
        'log_prediction_linear_reg',
        #'prediction',
        #'sales w-1'
       ]

def stacking_gen(train, test):
    df = pd.concat([train, test])
    test_dates = test.Date.sort_values().drop_duplicates()
    for d in test_dates:
        yield df[df.Date < d], df[df.Date == d]


preds = pd.DataFrame()
for train, test in stacking_gen(prediction_train, prediction_test):
    reg = LinearRegression()
    reg.fit(train[cols], train.target)
    test['ensemble'] = reg.predict(test[cols]) / sum(reg.coef_)
    preds = pd.concat([preds, test])

prediction_test['ensemble'] = reg.predict(prediction_test[cols]) / sum(reg.coef_)
prediction_test['ensemble'] = np.expm1(prediction_test.ensemble)

#from metrics.MAPE import MAPE
#print(MAPE(prediction_test[~prediction_test.target.isna()].real_target, prediction_test[~prediction_test.target.isna()].ensemble))

#prediction_test[['Date', 'sku', 'ensemble']].to_csv('../dataset/prediction/test/stacking.csv', index=False)