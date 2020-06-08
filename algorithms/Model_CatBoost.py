import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
sys.path.append('../')

from algorithms.Base_Model import BaseModel

class CatBoost(BaseModel):

    def __init__(self):
        super(CatBoost, self).__init__()

    def create(self, train, test, categorical_features=[], drop_columns=[], name='', isScope=True, sample_weights=None, evaluation=False):
        super().create(train=train, test=test, categorical_features=categorical_features, drop_columns=drop_columns,
                       name=name, isScope=isScope, sample_weights=sample_weights, evaluation=evaluation)

        self.params = {
                       # 'metric': 'huber',   # Se si cambia la metrica non si cambia l'ottimizzazione
                    #    'objective': CatBoost.wmape_train_,  # Per ottimizzare con una particolare metrica dobbiamo usare l'objective
                       'iterations': 600,
                        'learning_rate': 0.1,
                        #'depth': 4,
                        'verbose': False
                       }

        self.model = CatBoostRegressor(**self.params)

        return self

    def fit(self,):
        if self.evaluation:
            self.model.fit(self.X_train, self.y_train, cat_features=self.cat_features, sample_weight=self.sample_weights)
        else:
            self.model.fit(self.X_train, self.y_train, cat_features=self.cat_features, sample_weight=self.sample_weights)


    def predict(self,):
        self.X_test['log_prediction_' + self.name] = self.model.predict(self.X_test.drop(['target'] + self.drop_columns, axis=1))
        self.X_test['prediction_' + self.name] = np.expm1(self.X_test['log_prediction_' + self.name])

        return self.X_test[['Date', 'sku', 'target', 'real_target', 'log_prediction_' + self.name, 'prediction_' + self.name]]

    def run(self):
        if self.evaluation:
            self.fit()
        else:
            self.fit()
            return self.predict()

    def get_model(self):
        return self.model

    def plot_feature_importance(self, plot_title):
        plt.figure(figsize=(8, 5))
        plt.xticks(rotation=90)
        plt.title(plot_title)
        plt.plot(self.X_train.columns, self.model.get_feature_importance)
        plt.show()