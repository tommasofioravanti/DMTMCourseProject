import pandas as pd
import numpy as np

n_sku = None   # GLOBAL VARIABLE for the custom_wmape

class BaseModel(object):

    def __init__(self):
        pass

    def create(self, train, test, categorical_features=[], drop_columns=[], name='', isScope=True, sample_weights=None, evaluation=False):
        global n_sku

        if isScope:
            test = test[test.scope==1]

        self.drop_columns = drop_columns

        self.X_train = train.drop(['target'] + drop_columns, axis=1)
        self.y_train = train.target

        self.X_test = test.copy()
        #self.X_test = test.drop(['target'] + drop_columns, axis=1)
        self.y_test = test.target

        self.cat_features = categorical_features
        self.name = name
        self.sample_weights = sample_weights

        self.evaluation = evaluation

        n_sku = len(set(self.X_train.sku.values))

    def fit(self):
        """
        TO BE EXTENDED
        :return:
        """
        pass

    def predict(self):
        """
        TO BE EXTENDED
        :return:
        """
        pass

    def plot_importance(self):
        """
        TO BE EXTENDED
        :return:
        """
        pass

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
        global n_sku
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        N = int(y_true.shape[0] / n_sku)
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
        residual = (sum((abs(y_true - y_pred) / y_true) * (y_true / y_true_sum) * 100) / y_true.shape[0]).astype(
            "float")
        return "wmape", residual, False
