import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from tqdm import tqdm
from algorithms.Model_LightGBM import LightGBM
from algorithms.Model_CatBoost import CatBoost
from algorithms.Model_Linear_Regression import LinearRegressionClass
from algorithms.Model_Generator import Generator
import xgboost as xgb
import matplotlib.pyplot as plt

from preprocessing.preprocessing import preprocessing, train_validation_split
from metrics.MAPE import MAPE

from utils import add_all_features, dfs_gen

def run_xgboost(useTest=False, useScope=False, completeCV = False, dataAugm = False, save=True):

    train = pd.read_csv("../dataset/original/train.csv")
    test = pd.read_csv("../dataset/original/x_test.csv")

    useTest = useTest
    useScope = useScope
    isEvaluation = False
    useSampleWeights, weights_type = True, 2
    save = save

    completeCV = completeCV  # Per avere le predizioni sul train, impostarlo a True: parte dalla prima settimana del train
    # e predice via via tutte le settimane successive incrementando il train

    dataAugm = dataAugm  # Crea il 2016: consiglio di metterlo a True quando completeCV = True, in modo che l'algoritmo
    # non traini usando solo la prima settimana del train originale, ma usando tutto il 2016 [52 settimane]

    if isEvaluation:
        useTest = False
        useScope = False

    if completeCV:
        useTest = False
        useScope = False

    df = preprocessing(train, test, useTest=useTest, dataAugmentation=dataAugm)

    df, categorical_f = add_all_features(df)
    categorical_f = ['sku', 'pack', 'brand'] + categorical_f

    df = df.sort_values('Date')

    df_scope = df[['Date', 'sku', 'scope']].copy()

    def wmape_train_(y_true, data):
        """
        IMPORTANTE: sortare prima gli elementi del df per ('sku', 'Date'): df.sort_values(['sku','Date']

        Give less importance to previous [in time] values, exponentially
        :param y_true:
        :param y_pred:
        :return:
        """
        # global s
        y_true = np.array(y_true)
        y_pred = data.get_label()

        N = int(y_true.shape[0] / 133)
        weight = np.arange(y_true.shape[0])
        weight = weight % N
        weight = weight / N
        grad = -100 * ((y_true - y_pred) / y_true) * (np.exp(weight))
        hess = 100 / (y_true) * (np.exp(weight))
        return grad, hess

    def ohe_categorical(df, categorical_features):
        for c in categorical_features:
            dummy = pd.get_dummies(df[c], prefix=c)
            df[dummy.columns] = dummy
        return df

    df = ohe_categorical(df, ['cluster', 'heavy_light'])
    cat_cols = ['pack', 'brand', 'scope', 'heavy_light', 'cluster', 'year']
    df = df.drop(cat_cols, axis=1)

    if useTest:
        df = df.sort_values('Date')
        test_dates = df[df.Date >= '2019-06-29']
        test_dates = test_dates.drop_duplicates('Date').Date
        gen = dfs_gen(df, test_dates)
    else:
        train = df[~df.target.isna()]
        if completeCV:
            if dataAugm:
                dates = train[train.Date >= '2016-12-10'].Date.sort_values().drop_duplicates(keep='first')
            else:
                dates = train.Date.sort_values().drop_duplicates(keep='first')
            val_dates = dates[1:]
        else:
            _, _, val_dates = train_validation_split(train)
        gen = dfs_gen(train, val_dates)

    params = {
        'obj': wmape_train_,
        'learning_rate': 0.1,
        'max_depth': 10,
        # 'min_child_weight': 3,
        # 'tree_method': 'hist'
    }

    #  RUNNING MODEL

    prediction_df = pd.DataFrame()

    feature_importances = []

    prev_df_test = pd.DataFrame()
    drop_target = ['real_target', 'target', 'Date', 'sku']
    xgb_model = None
    for i, (df_train, df_test) in enumerate(gen):
        if i == 0:
            xgb_model = xgb.train(params, dtrain=xgb.DMatrix(df_train.drop(drop_target, axis=1), df_train.target),
                                  num_boost_round=700)

            feature_importances.append(xgb_model.get_fscore())

        else:
            # xgb_model.fit(prev_df_test.drop(drop_target, axis=1), prev_df_test.target, xgb_model='xgb_model_online.model')
            params.update({
                # 'learning_rate': 0.05,
                'updater': 'refresh',
                'process_type': 'update',
                'refresh_leaf': True,
                # 'reg_lambda': 3,  # L2
                # 'reg_alpha': 3,  # L1
                'silent': False,
            })

            xgb_model = xgb.train(params, dtrain=xgb.DMatrix(df_train.drop(drop_target, axis=1), df_train.target),
                                  num_boost_round=400,
                                  xgb_model=xgb_model)

        df_test['prediction'] = xgb_model.predict(xgb.DMatrix(df_test.drop(drop_target, axis=1)))
        # print(df_test[['Date', 'sku', 'target', 'prediction']])

        # xgb_model.save_model('xgb_model_online.model')
        prediction_df = pd.concat([prediction_df, df_test[['Date', 'sku', 'real_target', 'target', 'prediction']]])

        prev_df_test = df_test.drop(['prediction'], axis=1).copy()

    feature_importances.append(xgb_model.get_fscore())

    prediction_df['real_prediction'] = np.expm1(prediction_df.prediction)
    prediction_df = prediction_df.merge(df_scope, how='left', on=['Date', 'sku'])

    if not useTest:
        train = df[~df.target.isna()]
        _, _, val_dates = train_validation_split(train)
        mask_val = (prediction_df.Date.isin(val_dates)) & (prediction_df.scope == 1)
        print(f'MAPE {MAPE(prediction_df[mask_val].real_target, prediction_df[mask_val].real_prediction)}')

    if save:
        if useTest:
            prediction_df.drop('scope', axis=1).to_csv("../dataset/prediction/test/xgb_inc_test.csv", index=False)
        else:
            if completeCV:
                prediction_df.drop('scope', axis=1).to_csv("../dataset/prediction/val/xgb_inc_val.csv", index=False)

    plt.figure(figsize=(20, 10))

    feat_imp = {k: v for k, v in sorted(feature_importances[1].items(), key=lambda item: item[1])}

    x = list(feat_imp.keys())
    y = list(feat_imp.values())
    plt.barh(x, y)
    plt.show()


if __name__=='__main__':

    """
    ### Model
    Per far funzionare l'incremental learning, il primo modello che viene fatto partire deve essere un classico xgboost 
    con un certo numero di num_boost_rounds. I modelli successivi invece devono fare solo l'update del modello, non 
    devono creare nuovi alberi, dunque 'updater' viene impostato a 'refresh' e il 'process_type' = 'update'
    """

    # Running on Validation
    run_xgboost(useTest=False, useScope=False, completeCV = True, dataAugm=True, save=True)

    # Running on Test
    run_xgboost(useTest=True, useScope=True, completeCV=False, dataAugm=False, save=True)

