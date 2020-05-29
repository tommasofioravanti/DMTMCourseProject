import pandas as pd
import numpy as np
from tqdm import tqdm
from algorithms.Model_LightGBM import LGBM_Generator

import sys
#sys.path.append('.')

from preprocessing.preprocessing import preprocessing
from metrics.MAPE import MAPE

from utils import add_all_features

train = pd.read_csv("dataset/original/train.csv")
test = pd.read_csv("dataset/original/x_test.csv")

useTest = True
useScope = True
isEvaluation = False
useSampleWeights, weights_type = True, 0

if isEvaluation:
    useTest = False
    useScope = False
#   --------------- Preprocessing -----------------

df = preprocessing(train, test, useTest=useTest)

#   --------------- Features -----------------

df, categorical_f = add_all_features(df)
categorical_f = ['sku', 'pack', 'brand'] + categorical_f

df = df.sort_values('Date')
#   --------------- Model -----------------

drop_cols = ['scope', 'Date', 'real_target', 'pack', 'brand', 'size (GM)', 'cluster']
categorical_f = [x for x in categorical_f if x not in drop_cols]

standard_lgb = LGBM_Generator(df, categorical_features=categorical_f, drop_columns=drop_cols, isScope=useScope,
                     sample_weights_type=weights_type, evaluation=isEvaluation, useTest=useTest)

std_lgb_pred = standard_lgb.run_generator()

# ---- Predict by cluster  -----

#  -----------   Cluster 1

#drop_cols = drop_cols + ['cluster']
#categorical_f = [x for x in categorical_f if x not in drop_cols]

cluster_model_1 = LGBM_Generator(df, categorical_features=categorical_f, drop_columns=drop_cols, name='_c',
                                 isScope=useScope, cluster=1, useTest=useTest)

cluster_1_pred = cluster_model_1.run_generator()

#  -----------   Cluster 2

cluster_model_2 = LGBM_Generator(df, categorical_features=categorical_f, drop_columns=drop_cols, name='_c',
                                 isScope=useScope, cluster=2, useTest=useTest)

cluster_2_pred = cluster_model_2.run_generator()

cluster_pred = pd.concat([cluster_1_pred, cluster_2_pred])

if not useScope:
    #  -----------   Cluster 3

    cluster_model_3 = LGBM_Generator(df, categorical_features=categorical_f, drop_columns=drop_cols,
                                     name='_c',
                                     isScope=useScope, cluster=3, useTest=useTest)

    cluster_3_pred = cluster_model_3.run_generator()

    cluster_pred = pd.concat([cluster_pred, cluster_3_pred])



if not isEvaluation:
    # Merge Cluster Predictions with Standard LightGBM Predictions
    prediction_df = std_lgb_pred.merge(cluster_pred, how='left', on=['Date', 'sku', 'target', 'real_target'])

    print(prediction_df.columns)

    # Weighted Mean log_predictions
    res = []
    for l, l_c in tqdm(zip(prediction_df.log_prediction, prediction_df.log_prediction_c)):
        final_pred = (0.4 * l + 0.6 * l_c)
        res.append(final_pred)

    prediction_df['final_pred'] = np.expm1(res)


    if useTest:
        df_dates = prediction_df.sort_values('Date')['Date'].drop_duplicates()

        print(f'Standard MAPE = {MAPE(prediction_df[prediction_df.Date.isin(df_dates[:-1])].real_target,prediction_df[prediction_df.Date.isin(df_dates[:-1])].prediction)}')
        print(f'Cluster MAPE = {MAPE(prediction_df[prediction_df.Date.isin(df_dates[:-1])].real_target,prediction_df[prediction_df.Date.isin(df_dates[:-1])].prediction_c)}')
        print(f'Ensemble MAPE = {MAPE(prediction_df[prediction_df.Date.isin(df_dates[:-1])].real_target,prediction_df[prediction_df.Date.isin(df_dates[:-1])].final_pred)}')


    else:
        print(f'Standard MAPE = {MAPE(prediction_df.real_target, prediction_df.prediction)}')
        print(f'Cluster MAPE = {MAPE(prediction_df.real_target, prediction_df.prediction_c)}')
        print(f'Ensemble MAPE = {MAPE(prediction_df.real_target, prediction_df.final_pred)}')

    standard_lgb.plot_feature_importance('Standard')

    cluster_model_1.plot_feature_importance('Cluster 1')
    cluster_model_2.plot_feature_importance('Cluster 2')

    if not useScope:
        cluster_model_3.plot_feature_importance('Cluster 3')



