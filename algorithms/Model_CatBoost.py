import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

import sys
sys.path.append('.')

s = None        # Global variable to define the number of different skus in the train, used in the custom objective function

class CatBoost(object):

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
                    #    'objective': CatBoost.wmape_train_,  # Per ottimizzare con una particolare metrica dobbiamo usare l'objective
                       'iterations': 100,
                        'learning_rate': 0.2,
                        'depth': 4,
                        'verbose': False
                       }

        self.model = CatBoostRegressor(**self.params)

        self.name = name
        self.sample_weights = sample_weights

        self.evaluation = evaluation

        s = len(set(self.X_train.sku.values))

    def fit(self,):
        if self.evaluation:
            self.model.fit(self.X_train, self.y_train, cat_features=self.cat_features, sample_weight=self.sample_weights)
        else:
            self.model.fit(self.X_train, self.y_train, cat_features=self.cat_features, sample_weight=self.sample_weights)


    def predict(self,):
        self.X_test['log_prediction' + self.name] = self.model.predict(self.X_test.drop(['target'] + self.drop_columns, axis=1))
        self.X_test['prediction' + self.name] = np.expm1(self.X_test['log_prediction' + self.name])

        return self.X_test[['Date', 'sku', 'target', 'real_target', 'log_prediction' + self.name, 'prediction' + self.name]]

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