import pandas as pd
import numpy as np
from tqdm import tqdm

import os
import sys
sys.path.append('../')

from utils import dfs_gen, get_weights
from preprocessing.preprocessing import train_validation_split
from metrics.MAPE import MAPE

class Generator(object):

    def __init__(self, df, model, categorical_features, drop_columns, feature_subset=None, name='', isScope=True, sample_weights_type=None,
                 evaluation=False, cluster=None, useTest=True, completeCV=False, dataAugmentation=False):
        """

        :param df: complete dataframe [train + test]
        :param model: model to run
        :param categorical_features:
        :param drop_columns:
        :param name:
        :param isScope: bool, consider only scope == 1 or not for the evaluation
        :param sample_weights_type: type of sample weights
        :param evaluation: bool, if evaluation == True the model will be evaluated on the eval_set and no predictions are returned
        :param cluster: list, list of cluster to be considered
        :param useTest: predict on the original test set
        :param completeCV: start from prediction for the second week of the train, incrementally make prediction for
                            'all' the train set. Used for a second model step
        """
        self.useTest = useTest
        self.df = df
        if self.useTest:
            df = df.sort_values('Date')
            test_dates = df[df.Date >= '2019-06-29']
            test_dates = test_dates.drop_duplicates('Date').Date
            self.generator = dfs_gen(df, test_dates)
        else:
            train = df[~df.target.isna()]
            if completeCV:
                if dataAugmentation:
                    dates = train[train.Date >= '2016-12-10'].Date.sort_values().drop_duplicates(keep='first')
                else:
                    dates = train.Date.sort_values().drop_duplicates(keep='first')
                val_dates = dates[1:]
            else:
                _, _, val_dates = train_validation_split(train)
            self.generator = dfs_gen(train, val_dates)

        self.sample_weights_type = sample_weights_type
        self.sample_weights = None

        self.cat_features = categorical_features
        self.drop_columns = drop_columns
        self.feature_subset = feature_subset
        self.name = name
        self.isScope = isScope
        self.evaluation = evaluation
        self.predictions = pd.DataFrame()
        self.cluster = cluster
        self.model = model

        if cluster is not None:
            if isinstance(self.cluster, int):
                self.cluster = [self.cluster]
            elif isinstance(self.cluster, list):
                pass
            else:
                sys.exit("Cluster is not a list nor a integer")

        if self.evaluation:
            self.best_iterations = []


    def run_generator(self, save=False):

        for df_train, df_test in tqdm(self.generator):
            if self.feature_subset is not None:
                df_train = df_train[self.feature_subset+self.cat_features+self.drop_columns+['target']]
                df_test = df_test[self.feature_subset+self.cat_features+self.drop_columns+['target']]
            if self.sample_weights_type is not None:
                self.sample_weights = get_weights(df_train, type=self.sample_weights_type)

            if self.cluster is not None:
                for c in self.cluster:
                    self.model = self.model.create(df_train[df_train.cluster == c], df_test[df_test.cluster == c],
                                     categorical_features=self.cat_features,
                                     drop_columns=self.drop_columns, isScope=self.isScope,
                                     sample_weights=None, evaluation=self.evaluation, name=self.name)

                    if not self.evaluation:
                        self.predictions = pd.concat([self.predictions, self.model.run()])

            else:
                self.model = self.model.create(df_train, df_test, categorical_features=self.cat_features,
                                 drop_columns=self.drop_columns, isScope=self.isScope,
                                 sample_weights=self.sample_weights, evaluation=self.evaluation, name=self.name, )

                if not self.evaluation:
                    self.predictions = pd.concat([self.predictions, self.model.run()])

            if self.evaluation:
                self.best_iterations.append(self.model.run())
        if save:
            self.save_predictions()

        if self.evaluation:
            print(self.best_iterations)

        return self.predictions


    def save_predictions(self,):
        path = 'dataset/prediction'
        if not os.path.isdir(path):
            os.mkdir(path)

        if self.useTest:
            test_path = os.path.join(path, 'test')
            if not os.path.isdir(test_path):
                os.makedirs(test_path)
            self.predictions.to_csv(os.path.join(test_path, self.name + '_test.csv'), index=False)
        else:
            val_path = os.path.join(path, 'val')
            if not os.path.isdir(val_path):
                os.makedirs(val_path)
            self.predictions.to_csv(os.path.join(val_path, self.name + '_val.csv'), index=False)

    def plot_feature_importance(self, ):
        self.model.plot_feature_importance(self.name)

    def compute_MAPE(self):
        mape = None
        if not self.evaluation:
            if not self.useTest:
                _, _, val_dates = train_validation_split(self.df)
                mask_val_dates = (self.predictions.Date.isin(val_dates))
                mape = MAPE(self.predictions[mask_val_dates].real_target, self.predictions[mask_val_dates]["prediction_" + self.name])
                print(f'Standard MAPE = {mape}')

            return mape
