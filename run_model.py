import pandas as pd
import numpy as np
from tqdm import tqdm
from algorithms.Model_LightGBM import LightGBM
from algorithms.Model_CatBoost import CatBoost
from algorithms.Model_Linear_Regression import LinearRegressionClass
from algorithms.Model_Generator import Generator
from get_model_params import get_model_params

import sys
#sys.path.append('.')

from preprocessing.preprocessing import preprocessing, ohe_categorical
from metrics.MAPE import MAPE
from utils import add_all_features
from pathlib import Path
import os


def run_main(model_params, useTest=False, useScope=True, save=False, completeCV=False, dataAugm=True, drop_cols=[], cluster=None, name='', categorical_features=['sku', 'pack', 'brand']):

    abs_path = Path(__file__).absolute().parent
    train_path = os.path.join(abs_path, "dataset/original/train.csv")
    test_path = os.path.join(abs_path, "dataset/original/x_test.csv" )
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    useTest = useTest
    useScope = useScope
    isEvaluation = False
    useSampleWeights, weights_type = True, 2
    save = save
    completeCV = completeCV
    dataAugm = dataAugm

    if completeCV:
        useTest = False
        useScope = False

    df = preprocessing(train, test, useTest=useTest, dataAugmentation=dataAugm)

    df, categorical_f = add_all_features(df)

    categorical_f = list(set(categorical_features + categorical_f))
    drop_cols = drop_cols
    categorical_f = [x for x in categorical_f if x not in drop_cols]

    df = df.sort_values('Date')

    #   --------------- Model -----------------

    CLUSTER=cluster
    NAME=name

    if NAME == 'lgb_std' or NAME == 'lgb_cls':
        model = LightGBM(**model_params)

    elif NAME == 'catboost':
        model = CatBoost(**model_params)

    print('Start the model ' + NAME)
    model = model
    model_gen = Generator(df, model,
                          categorical_features=categorical_f,
                          drop_columns=drop_cols,
                          isScope=useScope,
                          sample_weights_type=weights_type,
                          evaluation=isEvaluation,
                          useTest=useTest,
                          cluster=CLUSTER,
                          name=NAME,
                          completeCV=completeCV,
                          dataAugmentation=dataAugm,
                          )

    model_gen.run_generator(save)
    model_gen.plot_feature_importance()
    print(model_gen.compute_MAPE())

def run_model(model_name, useTest):
    """

    :param model_name:    Select one of the following name:     - 'lgb_std'
                                                                - 'lgb_cls'
                                                                - 'catboost'

    :param useTest: True if you want to make prediction on Test set, False if you predict for Validation Set
    """

    model_params, params = get_model_params(model_name, useTest=useTest)
    print(params)

    run_main(model_params, **params)



if __name__=='__main__':

    run_model('lgb_std', useTest=False)


