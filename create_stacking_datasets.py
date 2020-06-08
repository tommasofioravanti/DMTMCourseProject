import pandas as pd
import numpy as np
from algorithms.Model_LightGBM import LightGBM
from algorithms.Model_CatBoost import CatBoost
from algorithms.Model_Linear_Regression import LinearRegressionClass

import sys
sys.path.append('../')

from main import main

categorical_features = ['cluster', 'sku', 'pack', 'brand'],
drop_cols = ['scope', 'Date', 'real_target', 'pack', 'size (GM)', 'cluster']

train_params = {
    'useTest':False,
    'useScope':False,
    'save':True,
    'completeCV':True,
    'dataAugm':True,
    'categorical_features':['cluster', 'sku', 'pack', 'brand'],
    'drop_cols':['scope', 'Date', 'real_target', 'pack', 'size (GM)', 'cluster'],
}

train_params_cluster = train_params.copy()
train_params_cluster['cluster'] = [1,2,3]

test_params = {
    'useTest':True,
    'useScope':True,
    'save':True,
    'completeCV':False,
    'dataAugm':False,
    'categorical_features':['cluster', 'sku', 'pack', 'brand'],
    'drop_cols':['scope', 'Date', 'real_target', 'pack', 'size (GM)', 'cluster'],
}

test_params_cluster = test_params.copy()
test_params_cluster['cluster'] = [1,2]

# Create Prediction on both train and Test

# LightGBM Standard
model = LightGBM()

train_params['name'] = 'lgb_std'
test_params['name'] = 'lgb_std'
#train_params['model'] = model
#test_params['model'] = model

#main(**train_params)
#main(**test_params)


# LightGBM Cluster
#model = LightGBM()

train_params_cluster['name'] = 'lgb_cls'
test_params_cluster['name'] = 'lgb_cls'
#train_params_cluster['model'] = model
#test_params_cluster['model'] = model

main(model = LightGBM(), **train_params_cluster)
#main(**test_params_cluster)



# Catboost Standard
model = CatBoost()

train_params['name'] = 'cat_std'
test_params['name'] = 'cat_std'

main(model=CatBoost(), **train_params)
#main(**test_params)


# Catboost Cluster

train_params_cluster['name'] = 'cat_cls'
test_params_cluster['name'] = 'cat_cls'

main(model=CatBoost(), **train_params_cluster)
#main(**test_params_cluster)


# Linear Regression per sku
model = LinearRegressionClass()

train_params['name'] = 'linear_reg'
test_params['name'] = 'linear_reg'

drop_cols = train_params['drop_cols'].copy()
drop_cols = drop_cols + ['pack', 'brand']

main(model=LinearRegressionClass(), **train_params)
#main(**test_params)
