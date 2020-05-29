import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from utils import dfs_gen, get_weights
from preprocessing.preprocessing import train_validation_split

s = None        # Global variable to define the number of different skus in the train, used in the custom objective function

class LightGBM(object):

    def __init__(self, train, test, categorical_features, drop_columns, name='', isScope=True, sample_weights=None, evaluation=False):
        global s

        if isScope:
            test = test[test.scope==1]

        self.drop_columns = drop_columns

        self.X_train = train.drop(['target'] + drop_columns, axis=1)
        self.y_train = train.target

        self.X_test = test.copy()
        #self.X_test = test.drop(['target'] + drop_columns, axis=1)
        self.y_test = test.target

        self.cat_features = categorical_features
        self.params = {
                       # 'metric': 'huber',   # Se si cambia la metrica non si cambia l'ottimizzazione
                       'objective': LightGBM.wmape_train_,  # Per ottimizzare con una particolare metrica dobbiamo usare l'objective
                       'verbose':-1,
                       'boosting_type':'gbdt',
                        'num_leaves':31,
                        'max_depth':- 1,
                        'learning_rate':0.1,
                       'n_estimators':600,
                       'min_split_gain':0.0,
                       'subsample':1.0,
                       'subsample_freq':0,
                       'colsample_bytree':1.0,
                       'reg_alpha':0.0,
                       'reg_lambda':0.0,
                       'random_state':None,
                       'silent':True,
                       'importance_type':'split',}

        self.model = LGBMRegressor(**self.params)

        self.name = name
        self.sample_weights = sample_weights

        self.evaluation = evaluation

        s = len(set(self.X_train.sku.values))

    def fit(self,):
        if self.evaluation:
            self.model.fit(self.X_train, self.y_train, categorical_feature=self.cat_features,
                           sample_weight=self.sample_weights, eval_set=[(self.X_test.drop(['target'] + self.drop_columns, axis=1), self.y_test)],
                           verbose=True, early_stopping_rounds=10, eval_metric=LightGBM.wmape_val_)
        else:
            self.model.fit(self.X_train, self.y_train, categorical_feature=self.cat_features, sample_weight=self.sample_weights)


    def predict(self,):
        self.X_test['log_prediction' + self.name] = self.model.predict(self.X_test.drop(['target'] + self.drop_columns, axis=1))
        self.X_test['prediction' + self.name] = np.expm1(self.X_test['log_prediction' + self.name])

        return self.X_test[['Date', 'sku', 'target', 'real_target', 'log_prediction' + self.name, 'prediction' + self.name]]

    #def  compute_mape(self):
    #    predictions = np.expm1(self.predictions)
    #    print(f'Day {d}     MAPE={MAPE(self.real_target, predictions)}')

    def plot_feature_importance(self, plot_title):
        plt.figure(figsize=(8, 5))
        plt.xticks(rotation=90)
        plt.title(plot_title)
        plt.plot(self.X_train.columns, self.model.feature_importances_)
        plt.show()

    def run(self):
        if self.evaluation:
            self.fit()
        else:
            self.fit()
            return self.predict()
        #self.compute_mape()
        #self.plot_feature_importance()

    def get_model(self):
        return self.model

    # Custom Objective Functions
    @staticmethod
    def wmape_train_(y_true, y_pred):
        """
        IMPORTANTE: sortare prima gli elementi del df per ('sku', 'Date'): df.sort_values(['sku','Date']

        Give less importance to previous [in time] values, exponentially
        :param y_true:
        :param y_pred:
        :return:
        """
        global s
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        N = int(y_true.shape[0] / s)
        weight = np.arange(y_true.shape[0])
        weight = weight % N
        weight = weight / N
        grad = -100 * ((y_true - y_pred) / y_true) * (np.exp(weight))
        hess = 100 / (y_true) * (np.exp(weight))

        # grad = -100 * ((y_true - y_pred) / y_true)
        # hess = 100 / (y_true)
        return grad, hess

    @staticmethod
    def huber_approx_obj(y_true, y_pred):
        d = y_pred - y_true
        h = 5  # h is delta in the graphic
        scale = 1 + (d / h) ** 2
        scale_sqrt = np.sqrt(scale)
        grad = d / scale_sqrt
        hess = 1 / scale / scale_sqrt
        return grad, hess

    @staticmethod
    def wmape_val_(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        y_true_sum = sum(y_true)
        residual = (sum((abs(y_true - y_pred) / y_true) * (y_true / y_true_sum) * 100) / y_true.shape[0]).astype("float")
        return "wmape", residual, False


"""
def wmape_train_(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residual = (y_true - y_pred).astype("float")

    #y_true_sum = sum(y_true)
    N = y_pred.shape[0]
    gradient = (1 / y_true) * 100 
    grad = np.where(residual < 0, gradient, -gradient)
    grad[residual == 0] = 0
    hess = 100/(y_true)
    return grad, hess
"""


"""
Ensemble MAPE = 9.386   Objective: default
Ensemble MAPE = 9.21    Objective: huber
Ensemble MAPE = 10.789  Objective: mape
"""

class LGBM_Generator(object):

    def __init__(self, df, categorical_features, drop_columns, name='', isScope=True, sample_weights_type=None,
                 evaluation=False, cluster=None, useTest=True):

        if useTest:
            df = df.sort_values('Date')
            test_dates = df[df.Date >= '2019-06-29']
            test_dates = test_dates.drop_duplicates('Date').Date
            self.generator = dfs_gen(df, test_dates)
        else:
            train = df[~df.target.isna()]
            _, _, val_dates = train_validation_split(train)
            self.generator = dfs_gen(train, val_dates)

        self.sample_weights_type = sample_weights_type
        self.sample_weights = None

        self.cat_features = categorical_features
        self.drop_columns = drop_columns
        self.name = name
        self.isScope = isScope
        self.evaluation = evaluation
        self.predictions = pd.DataFrame()
        self.cluster = cluster

    def run_generator(self):

        for df_train, df_test in tqdm(self.generator):

            if self.sample_weights_type is not None:
                self.sample_weights = get_weights(df_train, type=self.sample_weights_type)

            if self.cluster is not None:
                self.model = LightGBM(df_train[df_train.cluster == self.cluster], df_test[df_test.cluster == self.cluster],
                                 categorical_features=self.cat_features,
                                 drop_columns=self.drop_columns, isScope=self.isScope,
                                 sample_weights=self.sample_weights, evaluation=self.evaluation, name=self.name)
            else:
                self.model = LightGBM(df_train, df_test, categorical_features=self.cat_features,
                                 drop_columns=self.drop_columns, isScope=self.isScope,
                                 sample_weights=self.sample_weights, evaluation=self.evaluation, name=self.name, )

            if not self.evaluation:
                self.predictions = pd.concat([self.predictions, self.model.run()])

        #self.predictions.to_csv(f"prediction_{self.name}.csv", index=False)
        return self.predictions

    def plot_feature_importance(self, plot_title):
        self.model.plot_feature_importance(self.name)
