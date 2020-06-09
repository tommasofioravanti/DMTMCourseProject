import pandas as pd
import numpy as np
from tqdm import tqdm
from algorithms.Model_LightGBM import LightGBM
from algorithms.Model_CatBoost import CatBoost
from algorithms.Model_Linear_Regression import LinearRegressionClass
from algorithms.Model_Generator import Generator

import sys
#sys.path.append('.')

from preprocessing.preprocessing import preprocessing, ohe_categorical
from metrics.MAPE import MAPE

from utils import add_all_features


def main(model, useTest, useScope, save, completeCV, dataAugm, categorical_features = ['cluster', 'sku', 'pack', 'brand'],
         drop_cols = ['scope', 'Date', 'real_target', 'pack', 'size (GM)', 'cluster'], cluster=None, name='',
         useSampleWeights=True, weights_type = 2, isEvaluation = False, rand_noise=False):

    train = pd.read_csv("dataset/original/train.csv")
    test = pd.read_csv("dataset/original/x_test.csv")

    useTest = useTest
    useScope = useScope
    isEvaluation = isEvaluation
    useSampleWeights, weights_type = useSampleWeights, weights_type
    save = save

    completeCV = completeCV  # Per avere le predizioni sul train, impostarlo a True: parte dalla prima settimana del train
    # e predice via via tutte le settimane successive incrementando il train

    dataAugm = dataAugm  # Crea il 2016: consiglio di metterlo a True quando completeCV = True, in modo che l'algoritmo
    # non traini usando solo la prima settimana del train originale, ma usando tutto il 2016 [52 settimane]
    rand_noise = rand_noise

    if isEvaluation:
        useTest = False
        useScope = False

    if completeCV:
        useTest = False
        useScope = False

    df = preprocessing(train, test, useTest=useTest, dataAugmentation=dataAugm, rand_noise=rand_noise)

    df, categorical_f = add_all_features(df)
    #categorical_f = ['sku', 'pack', 'brand'] + categorical_f
    categorical_f = categorical_features
    drop_cols = drop_cols

    df = df.sort_values('Date')

    #   --------------- Model -----------------

    #drop_cols = ['scope', 'Date', 'real_target', 'pack', 'size (GM)', 'cluster']
    categorical_f = [x for x in categorical_f if x not in drop_cols]

    # df = ohe_categorical(df, [c for c in categorical_f if c != 'sku'])    # Usare per mettere in ohe le features categoriche

    # CLUSTER = [1,2,3]      # Set CLUSTER = None if you want NOT to consider any cluster
    CLUSTER = cluster
    NAME = name

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

    print(model_gen.compute_MAPE())