import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from algorithms.Base_Model import BaseModel


class LinearRegressionClass(BaseModel):

    def __init__(self):
        super(LinearRegressionClass, self).__init__()

    def create(self, train, test, categorical_features=[], drop_columns=[], name='', isScope=True, sample_weights=None, evaluation=False):
        super().create(train=train, test=test, categorical_features=categorical_features, drop_columns=drop_columns,
                       name=name, isScope=isScope, sample_weights=sample_weights, evaluation=evaluation)

        self.model = LinearRegression()
        return self


    def fit(self,):
        print(self.X_train_tmp.columns)
        self.model.fit(self.X_train_tmp, self.y_train_tmp)


    def predict(self,):
        self.X_test_tmp['log_prediction_' + self.name] = self.model.predict(self.X_test_tmp.drop(['target','sku'] + self.drop_columns, axis=1))
        self.X_test_tmp['prediction_' + self.name] = np.expm1(self.X_test_tmp['log_prediction_' + self.name])

        return self.X_test_tmp[['Date', 'sku', 'target', 'real_target', 'log_prediction_' + self.name, 'prediction_' + self.name]]

    def plot_feature_importance(self):
        print(self.model.coef_)

    def run(self):
        self.X_train = self.X_train.fillna(0)
        self.X_test = self.X_test.fillna(0)
        predictions = pd.DataFrame()
        if self.evaluation:
            print('No Evaluation for Linear Regression')
        else:
            for s in set(self.X_test.sku):
                mask_train = self.X_train.sku == s
                mask_test = self.X_test.sku == s
                self.X_train_tmp = self.X_train[mask_train].drop('sku', axis=1).copy()
                self.y_train_tmp = self.y_train.loc[self.X_train_tmp.index]
                self.X_test_tmp = self.X_test[mask_test].copy()
                print(self.X_test_tmp.columns)

                self.fit()
                predictions = pd.concat([predictions, self.predict()])
        return predictions

    def get_model(self):
        return self.model
