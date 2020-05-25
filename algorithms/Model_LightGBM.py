import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

import sys
sys.path.append('.')

class LightGBM(object):

    def __init__(self, train, test, categorical_features, drop_columns):

        test = test[test.scope==1]
        self.drop_columns = drop_columns
        self.df_target = test[['Date', 'sku', 'target', 'real_target']]

        self.X_train = train.drop(['target'] + drop_columns, axis=1)
        self.y_train = train.target

        self.X_test = test
        #self.X_test = test.drop(['target'] + drop_columns, axis=1)
        self.y_test = test.target

        self.cat_features = categorical_features
        self.params = {'metric': 'mape'}

        self.model = LGBMRegressor(**self.params)


    def fit(self,):
        self.model.fit(self.X_train, self.y_train, categorical_feature=self.cat_features)

    def predict(self,):
        self.X_test['log_prediction'] = self.model.predict(self.X_test.drop(['target'] + self.drop_columns, axis=1))
        self.df_target = self.df_target.merge(self.X_test[['Date', 'sku', 'log_prediction']], how='left', on=['Date', 'sku'])

        self.df_target['prediction'] = np.expm1(self.df_target.log_prediction)
        return self.df_target

    #def  compute_mape(self):
    #    predictions = np.expm1(self.predictions)
    #    print(f'Day {d}     MAPE={MAPE(self.real_target, predictions)}')

    def plot_feature_importance(self):
        plt.figure(figsize=(8, 5))
        plt.xticks(rotation=90)
        plt.plot(self.X_train.columns, self.model.feature_importances_)
        plt.show()

    def run(self):
        self.fit()
        return self.predict()
        #self.compute_mape()
        #self.plot_feature_importance()
