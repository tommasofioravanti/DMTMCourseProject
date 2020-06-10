from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import forest_minimize
import pandas as pd
import numpy as np
from tqdm import tqdm
from algorithms.Model_LightGBM import LightGBM
from algorithms.Model_CatBoost import CatBoost
from algorithms.Model_Linear_Regression import LinearRegressionClass
from algorithms.Model_Generator import Generator

import sys
#sys.path.append('.')

from preprocessing.preprocessing import preprocessing
from metrics.MAPE import MAPE

from utils import add_all_features

train = pd.read_csv("dataset/original/train.csv")
test = pd.read_csv("dataset/original/x_test.csv")

useTest = False
useScope = True
isEvaluation = False
useSampleWeights, weights_type = True, 2
save = False

completeCV = False      # Per avere le predizioni sul train, impostarlo a True: parte dalla prima settimana del train
                        # e predice via via tutte le settimane successive incrementando il train

dataAugm = True        # Crea il 2016: consiglio di metterlo a True quando completeCV = True, in modo che l'algoritmo
                        # non traini usando solo la prima settimana del train originale, ma usando tutto il 2016 [52 settimane]

if isEvaluation:
    useTest = False
    useScope = False

if completeCV:
    useTest = False
    useScope = False

df = preprocessing(train, test, useTest=useTest, dataAugmentation=dataAugm)

df, categorical_f = add_all_features(df)
categorical_f = ['sku', 'pack', 'brand'] + categorical_f

df = df.sort_values('Date')

#   --------------- Model -----------------

drop_cols = ['scope', 'Date', 'real_target', 'pack', 'size (GM)', 'cluster']
categorical_f = [x for x in categorical_f if x not in drop_cols]
# feature_subset = [
#     'sales w-1',
#     'price',
#     'week_of_the_year',
#     'lag_target_50',
#     'increment',
#     'heavy_light',
#     'volume_on_promo w-1',
#     'seasons',
#     'Corr'
#     ]
feature_subset = None
#CLUSTER = [1,2,3]      # Set CLUSTER = None if you want NOT to consider any cluster
CLUSTER = None
NAME = 'lightgbm'

space  = [
    Integer(1, 20, name='max_depth'),
    Real(10**-2, 10**0-0.5, "log-uniform", name='learning_rate'),
    Integer(400, 1000, name='n_estimators'),
    Integer(3,50,name='num_leaves')
]
# 13.89245504314256 
# con
# params = {
        #                # 'metric': 'huber',   # Se si cambia la metrica non si cambia l'ottimizzazione
        #                'verbose':-1,
        #                'boosting_type':'gbdt',
        #                 'num_leaves':31,
        #                 'max_depth':- 1,
        #                 'learning_rate':0.1,
        #                'n_estimators':600,
        #                'min_split_gain':0.0,
        #                'subsample':1.0,
        #                'subsample_freq':0,
        #                'colsample_bytree':1.0,
        #                'reg_alpha':0.0,
        #                'reg_lambda':0.0,
        #                'random_state':None,
        #                'silent':True,
        #                'importance_type':'split',
        #                 'tree_learner':'feature',
        # }
@use_named_args(space)
def objective(**params):
    model = LightGBM(**params)
    model_gen = Generator(df, model,
                            categorical_features=categorical_f,
                            drop_columns=drop_cols,
                            feature_subset=feature_subset,
                            isScope=useScope,
                            sample_weights_type=weights_type,
                            evaluation=isEvaluation,
                            useTest=useTest,
                            cluster=CLUSTER,
                            name=NAME,
                            completeCV=completeCV,
                            dataAugmentation=dataAugm,
                            )

    prediction = model_gen.run_generator(save)
    return model_gen.compute_MAPE()

res_gp = forest_minimize(objective, space, n_calls=50, random_state=17, verbose=False)

print("Best score=%.4f" % res_gp.fun)
print("""Best parameters:
- max_depth=%d
- learning_rate=%.6f
- n_estimators=%d
- num_leaver=%d""" 
    % (res_gp.x[0], res_gp.x[1], 
    res_gp.x[2], res_gp.x[3]))