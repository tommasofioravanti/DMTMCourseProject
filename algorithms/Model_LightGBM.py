import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

import sys
sys.path.append('.')

class LightGBM(object):

    def __init__(self, train, test, categorical_features, drop_columns, name='', isScope=True, sample_weights=None):

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
        self.params = {'metric': 'mape',
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

    def fit(self,):
        self.model.fit(self.X_train, self.y_train, categorical_feature=self.cat_features, sample_weight=self.sample_weights)


    def predict(self,):
        self.X_test['log_prediction' + self.name] = self.model.predict(self.X_test.drop(['target'] + self.drop_columns, axis=1))
        self.X_test['prediction' + self.name] = np.expm1(self.X_test['log_prediction'+ self.name])

        return self.X_test[['Date', 'sku', 'target', 'real_target' ,'log_prediction'+ self.name, 'prediction'+ self.name]]

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
        self.fit()
        return self.predict()
        #self.compute_mape()
        #self.plot_feature_importance()
