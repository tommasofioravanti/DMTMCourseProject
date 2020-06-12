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

# Create Prediction on both Train and Test
"""
# LightGBM Standard
lgb_model_params = {'num_leaves':31, 'max_depth': -1, 'learning_rate': 0.06, 'n_estimators': 950}

lgb_train_params = train_params.copy()
lgb_test_params = test_params.copy()

lgb_train_params['name'] = 'lgb_std'
lgb_test_params['name'] = 'lgb_std'

lgb_train_params['drop_cols'] = ['scope',
                                 'Date',
                                 'real_target',
                                 'pack',
                                 'size (GM)',
                                 'cluster',
                                 'brand',
                                 'week_of_the_year',
                                 'year']

lgb_test_params['drop_cols'] = ['scope',
                                'Date',
                                'real_target',
                                'pack',
                                'size (GM)',
                                'cluster',
                                'brand',
                                'week_of_the_year',
                                'year']

main(model=LightGBM(**lgb_model_params),**lgb_train_params)
main(model=LightGBM(**lgb_model_params),**lgb_test_params)

# LightGBM Cluster
lgb_cluster_model_params = {'num_leaves':31, 'max_depth': -1, 'learning_rate': 0.06, 'n_estimators': 650}

lgb_train_params_cluster = train_params_cluster.copy()
lgb_test_params_cluster = test_params_cluster.copy()

lgb_train_params_cluster['name'] = 'lgb_cls'
lgb_test_params_cluster['name'] = 'lgb_cls'

lgb_test_params_cluster['dataAugm'] = True

lgb_train_params_cluster['drop_cols'] = ['scope', 'Date', 'real_target', 'pack', 'size (GM)', 'cluster','brand', 'week_of_the_year', 'year']
lgb_test_params_cluster['drop_cols'] = ['scope', 'Date', 'real_target', 'pack', 'size (GM)', 'cluster','brand', 'week_of_the_year', 'year']

main(model = LightGBM(**lgb_cluster_model_params), **lgb_train_params_cluster)
main(model=LightGBM(**lgb_cluster_model_params),**lgb_test_params_cluster)
"""

# Catboost Standard
catboost_model_params = {'num_leaves':31, 'learning_rate':0.1, 'n_estimators':600}

catboost_train_params = train_params.copy()
catboost_test_params = test_params.copy()

catboost_train_params['name'] = 'catboost'
catboost_test_params['name'] = 'catboost'

gte_cols = ['gte_pack','gte_brand','gte_cluster','gte_pack_brand',
            'gte_pack_cluster','gte_brand_cluster','gte_pack_brand_cluster']

catboost_train_params['drop_cols'] = ['scope', 'Date', 'real_target', 'pack', 'size (GM)', 'cluster'] + gte_cols
catboost_test_params['drop_cols'] = ['scope', 'Date', 'real_target', 'pack', 'size (GM)', 'cluster'] + gte_cols

main(model=CatBoost(**catboost_model_params), **catboost_train_params)
main(model=CatBoost(**catboost_model_params),**catboost_test_params)

"""
# Linear Regression per sku
train_params['name'] = 'linear_reg'
test_params['name'] = 'linear_reg'

drop_cols = train_params['drop_cols'].copy()
drop_cols = drop_cols + ['pack', 'brand']

train_params['drop_cols'] = drop_cols
test_params['drop_cols'] = drop_cols

train_params['rand_noise'] = True
test_params['rand_noise'] = True

main(model=LinearRegressionClass(), **train_params)
main(model=LinearRegressionClass(), **test_params)
"""