from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt


class RandomForest(object):

    def __init__(self, train, test, drop_columns, name='', isScope=True, sample_weights=None):
        if isScope:
            test = test[test.scope == 1]

        self.drop_columns = drop_columns + ['sku']

        self.X_train = train.drop(['target'] + self.drop_columns, axis=1)
        self.y_train = train.target

        self.X_test = test.copy()

        self.y_test = test.target

        self.params = {
            'n_estimators': 1000,
            # 'random_state':1,
            'criterion': "mse",
            'bootstrap': True,
            #'max_depth':5

        }

        self.model = RandomForestRegressor(**self.params)

        self.name = name
        self.sample_weights = sample_weights

    def fit(self, ):
        self.model.fit(self.X_train, self.y_train, sample_weight=self.sample_weights)

    def predict(self, ):
        self.X_test['log_prediction' + self.name] = self.model.predict(
            self.X_test.drop(['target'] + self.drop_columns, axis=1))
        self.X_test['prediction' + self.name] = np.expm1(self.X_test['log_prediction' + self.name])

        return self.X_test[
            ['Date', 'sku', 'target', 'real_target', 'log_prediction' + self.name, 'prediction' + self.name]]


    def plot_feature_importance(self, plot_title):
        plt.figure(figsize=(8, 5))
        plt.xticks(rotation=90)
        plt.title(plot_title)
        plt.plot(self.X_train.columns, self.model.feature_importances_)
        plt.show()

    def run(self):
        self.fit()
        return self.predict()

