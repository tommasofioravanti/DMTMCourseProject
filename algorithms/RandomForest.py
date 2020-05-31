from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

from algorithms.Base_Model import BaseModel


class RandomForest(BaseModel):
    
    def __init__(self):
        super(RandomForest, self).__init__()

    def create(self, train, test, categorical_features=[], drop_columns=[], name='', isScope=True, sample_weights=None, evaluation=False):
        super().create(train=train, test=test, categorical_features=categorical_features, drop_columns=drop_columns,
                       name=name, isScope=isScope, sample_weights=sample_weights, evaluation=evaluation)

        self.params = {
            'n_estimators': 1000,
            # 'random_state':1,
            'criterion': "mse",
            'bootstrap': True,
            #'max_depth':5

        }

        self.model = RandomForestRegressor(**self.params)
        return self

    def fit(self, ):
        self.model.fit(self.X_train, self.y_train, sample_weight=self.sample_weights)

    def predict(self, ):
        self.X_test['log_prediction_' + self.name] = self.model.predict(self.X_test.drop(['target'] + self.drop_columns, axis=1))
        self.X_test['prediction_' + self.name] = np.expm1(self.X_test['log_prediction_' + self.name])

        return self.X_test[
            ['Date', 'sku', 'target', 'real_target', 'log_prediction_' + self.name, 'prediction_' + self.name]]


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

