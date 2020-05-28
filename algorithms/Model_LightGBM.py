import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

import sys
sys.path.append('.')

class LightGBM(object):

    def __init__(self, train, test, categorical_features, drop_columns, name='', isScope=True, sample_weights=None, evaluation=False):

        if isScope:
            test = test[test.scope==1]

        self.drop_columns = drop_columns
        self.df_target = test[['Date', 'sku', 'target', 'real_target']]

        self.X_train = train.drop(['target'] + drop_columns, axis=1)
        self.y_train = train.target

        self.X_test = test.copy()
        #self.X_test = test.drop(['target'] + drop_columns, axis=1)
        self.y_test = test.target


        self.cat_features = categorical_features
        self.params = {
                       # 'metric': 'huber',   # Se si cambia la metrica non si cambia l'ottimizzazione
                       #'objective': 'mape',  # Per ottimizzare con una particolare metrica dobbiamo usare l'objective
                       'verbose':-1,
                       'boosting_type':'gbdt',
                        'num_leaves':31,
                        'max_depth':- 1,
                        'learning_rate':0.1,
                       'n_estimators':100,
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

    def fit(self,):
        if self.evaluation:
            self.model.fit(self.X_train, self.y_train, categorical_feature=self.cat_features,
                           sample_weight=self.sample_weights, eval_set=[(self.X_test.drop(['target'] + self.drop_columns, axis=1), self.y_test)],
                           verbose=True, early_stopping_rounds=10, eval_metric=wmape_val_)
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


def wmape_train_(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    grad = -100*((y_true - y_pred)/y_true)
    hess = 100/(y_true)
    return grad, hess

def huber_approx_obj(y_true, y_pred):
    d = y_pred - y_true
    h = 5  #h is delta in the graphic
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess


"""
Ensemble MAPE = 9.386   Objective: default
Ensemble MAPE = 9.21    Objective: huber
Ensemble MAPE = 10.789  Objective: mape
"""