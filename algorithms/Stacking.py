import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
sys.path.append('../')
from preprocessing.preprocessing import preprocessing, train_validation_split
from metrics.MAPE import MAPE

import warnings
warnings.filterwarnings("ignore")

path_train = '../dataset/prediction/val/'
path_test = '../dataset/prediction/test/'

# Load predictions
# LightGBM standard
lgb_std_train = pd.read_csv(path_train + "lgb_std_val.csv")
lgb_std_test = pd.read_csv(path_test + "lgb_std_test.csv")

# LightGBM cluster
lgb_cls_train = pd.read_csv(path_train + "lgb_cls_val.csv")
lgb_cls_test = pd.read_csv(path_test + "lgb_cls_test.csv")

# Catboost standard
cat_std_train = pd.read_csv(path_train + "catboost_val.csv")
cat_std_test = pd.read_csv(path_test + "catboost_test.csv")

# XGBoost Incremental
xgb_train = pd.read_csv(path_train + "xgb_inc_val.csv")
xgb_test = pd.read_csv(path_test + "xgb_inc_test.csv")

train = pd.read_csv("../dataset/original/train.csv")
test = pd.read_csv("../dataset/original/x_test.csv")
df = preprocessing(train, test, useTest=False)

df_scope = df[['Date', 'sku', 'scope']].copy()

# Train
prediction_train = cat_std_train.merge(lgb_std_train, how='left', on=['Date', 'sku', 'target', 'real_target'])
prediction_train = lgb_cls_train.merge(prediction_train, how='left', on=['Date', 'sku', 'target', 'real_target'])
prediction_train = prediction_train.merge(xgb_train, how='left', on=['Date', 'sku', 'target', 'real_target'])

prediction_train.Date = pd.to_datetime(prediction_train.Date)

prediction_train.Date = pd.to_datetime(prediction_train.Date)
prediction_train = prediction_train.merge(df[['Date', 'sku', 'scope']], how='left', on=['Date', 'sku'])

# Test
prediction_test = cat_std_test.merge(lgb_std_test, how='left', on=['Date', 'sku', 'target', 'real_target'])
prediction_test = lgb_cls_test.merge(prediction_test, how='left', on=['Date', 'sku', 'target', 'real_target'])
prediction_test = prediction_test.merge(xgb_test, how='left', on=['Date', 'sku', 'target', 'real_target'])

prediction_test.Date = pd.to_datetime(prediction_test.Date)
#prediction_test = prediction_test.merge(df[['Date', 'sku', 'scope']], how='left')
#prediction_test = prediction_test.merge(gte, how='left', on=['Date', 'sku', 'target', 'real_target'])


# Take only 2017 since 2016 has been augmented
#mask = (prediction_train.Date<='2017-01-07')
#prediction_train = prediction_train.drop(prediction_train[mask].index)


cols = ['log_prediction_lgb_cls',
        'log_prediction_lgb_std',
        'log_prediction_catboost',
        'prediction',
       ]


# Validation Prediction
def stacking_gen_val(train):
    _, _, val_dates = train_validation_split(train)
    train = train.sort_values('Date').reset_index(drop=True)
    for d in val_dates:
        yield train[train.Date < d], train[train.Date == d]


preds_val = pd.DataFrame()
for train, test in stacking_gen_val(prediction_train):
    reg = LinearRegression()
    reg.fit(train[cols], train.target)
    test['ensemble'] = reg.predict(test[cols]) / sum(reg.coef_)
    preds_val = pd.concat([preds_val, test])

preds_val['ensemble'] = np.expm1(preds_val.ensemble)
#print(f'MAPE on validation set: {MAPE(preds_val.real_target, preds_val.ensemble)}')

train = pd.read_csv("../dataset/original/train.csv")
test = pd.read_csv("../dataset/original/x_test.csv")
df = preprocessing(train, test, useTest=False, dataAugmentation=True)
_, _, val_dates = train_validation_split(df[~df.target.isna()])

mask_val_dates = (prediction_train.Date.isin(val_dates)) & (prediction_train.scope==1)
print(f'MAPE LightGBM Standard on validation set: {MAPE(prediction_train[mask_val_dates].real_target, prediction_train[mask_val_dates].prediction_lgb_std)}')
print(f'MAPE LightGBM Cluster on validation set: {MAPE(prediction_train[mask_val_dates].real_target, prediction_train[mask_val_dates].prediction_lgb_cls)}')
print(f'MAPE Catboost Standard on validation set: {MAPE(prediction_train[mask_val_dates].real_target, prediction_train[mask_val_dates].prediction_catboost)}')
print(f'MAPE XGBoost on validation set: {MAPE(prediction_train[mask_val_dates].real_target, prediction_train[mask_val_dates].real_prediction)}')

#preds_val = preds_val.merge(df_scope, how='left', on=['Date', 'sku'])
print(preds_val.columns)
preds_val_mask = (preds_val.Date.isin(val_dates)) & (preds_val.scope==1)
print(f'MAPE on validation set: {MAPE(preds_val[preds_val_mask].real_target, preds_val[preds_val_mask].ensemble)}')


# Test Prediction
def stacking_gen_test(train, test):
    df = pd.concat([train, test])
    test_dates = test.Date.sort_values().drop_duplicates()
    for d in test_dates:
        yield df[df.Date < d], df[df.Date == d]


preds = pd.DataFrame()
for train, test in stacking_gen_test(prediction_train, prediction_test):
    reg = LinearRegression()
    reg.fit(train[cols], train.target)
    test['ensemble'] = reg.predict(test[cols]) / sum(reg.coef_)
    preds = pd.concat([preds, test])

preds['ensemble'] = np.expm1(preds.ensemble)


def extract_subm(stack_pred):
    # stack_pred['Date'] = pd.to_datetime(stack_pred['Date'])
    subm = pd.read_csv("../dataset/original/example_submission.csv")
    subm = subm.drop('prediction', axis=1)
    from preprocessing.preprocessing import convert_date
    subm['Date'] = convert_date(subm[['Unnamed: 0']])
    subm = subm.merge(stack_pred, how='left', on=['Date', 'sku']).rename(columns={'ensemble':'prediction'}).set_index('Unnamed: 0')
    return subm.drop('Date', axis=1)


preds = extract_subm(preds[['Date', 'sku', 'ensemble']])
print(preds.columns)


preds.to_csv('../dataset/prediction/test/stacking_preds.csv')





print(list(zip(cols, reg.coef_)))
