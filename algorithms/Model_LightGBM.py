import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from algorithms.Base_Model import BaseModel


class LightGBM(BaseModel):

    def __init__(self):
        super(LightGBM, self).__init__()

    def create(self, train, test, categorical_features=[], drop_columns=[], name='', isScope=True, sample_weights=None, evaluation=False):
        super().create(train=train, test=test, categorical_features=categorical_features, drop_columns=drop_columns,
                       name=name, isScope=isScope, sample_weights=sample_weights, evaluation=evaluation)

        self.params = {
                       # 'metric': 'huber',   # Se si cambia la metrica non si cambia l'ottimizzazione
                       'objective': BaseModel.wmape_train_,  # Per ottimizzare con una particolare metrica dobbiamo usare l'objective
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
                       'importance_type':'split',
                        'tree_learner':'feature',
        }

        self.model = LGBMRegressor(**self.params)
        return self


    def fit(self,):

        if self.evaluation:
            self.model.fit(self.X_train, self.y_train, categorical_feature=self.cat_features,
                           sample_weight=self.sample_weights, eval_set=[(self.X_test.drop(['target'] + self.drop_columns, axis=1), self.y_test)],
                           verbose=False, early_stopping_rounds=50, eval_metric=LightGBM.wmape_val_, )
            return self.model.best_iteration_
        else:
            self.model.fit(self.X_train, self.y_train, categorical_feature=self.cat_features, sample_weight=self.sample_weights)


    def predict(self,):
        self.X_test['log_prediction_' + self.name] = self.model.predict(self.X_test.drop(['target'] + self.drop_columns, axis=1))
        self.X_test['prediction_' + self.name] = np.expm1(self.X_test['log_prediction_' + self.name])

        return self.X_test[['Date', 'sku', 'target', 'real_target', 'log_prediction_' + self.name, 'prediction_' + self.name]]

    def plot_feature_importance(self, plot_title):
        plt.figure(figsize=(8, 5))
        plt.xticks(rotation=90)
        plt.title(plot_title)
        plt.plot(self.X_train.columns, self.model.feature_importances_)
        plt.show()

    def run(self):
        if self.evaluation:
            return self.fit()
        else:
            self.fit()
            return self.predict()

    def get_model(self):
        return self.model


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

