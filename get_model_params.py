
def get_model_params(model_name, useTest):

    if model_name == 'lgb_std':
        model_params = {'boosting_type':'gbdt',
                        'num_leaves':31,
                        'max_depth':- 1,
                        'learning_rate':0.06,
                       'n_estimators':950,}
        if useTest:
            params = {'useTest':True,
                      'useScope':True,
                      'save':False,
                      'completeCV':False,
                      'dataAugm':False,
                      'drop_cols':['scope', 'Date', 'real_target', 'pack', 'size (GM)', 'cluster','brand', 'week_of_the_year', 'year'],
                      'cluster':None,
                      'name':'lgb_std'
                      }
        else:
            params = {'useTest': False,
                      'useScope': True,
                      'save': False,
                      'completeCV': False,
                      'dataAugm': True,
                      'drop_cols': ['scope', 'Date', 'real_target', 'pack', 'size (GM)', 'cluster', 'brand',
                                    'week_of_the_year', 'year'],
                      'cluster': None,
                      'name': 'lgb_std'
                      }
        return model_params, params


    elif model_name == 'lgb_cls':
        model_params = {'boosting_type': 'gbdt',
                        'num_leaves': 31,
                        'max_depth': - 1,
                        'learning_rate': 0.06,
                        'n_estimators': 650, }
        if useTest:
            params = {'useTest': True,
                      'useScope': True,
                      'save': False,
                      'completeCV': False,
                      'dataAugm': True,
                      'drop_cols': ['scope', 'Date', 'real_target', 'pack', 'size (GM)', 'cluster', 'brand',
                                    'week_of_the_year', 'year'],
                      'cluster': [1,2],
                      'name': 'lgb_cls'
                      }
        else:
            params = {'useTest': False,
                      'useScope': True,
                      'save': False,
                      'completeCV': False,
                      'dataAugm': True,
                      'drop_cols': ['scope', 'Date', 'real_target', 'pack', 'size (GM)', 'cluster', 'brand',
                                    'week_of_the_year', 'year'],
                      'cluster': [1,2],
                      'name': 'lgb_cls'
                      }
        return model_params, params

    elif model_name == 'catboost':
        model_params = {
                        'learning_rate': 0.1,
                        'n_estimators': 600, }

        gte_cols = ['gte_pack', 'gte_brand', 'gte_cluster', 'gte_pack_brand',
                    'gte_pack_cluster', 'gte_brand_cluster', 'gte_pack_brand_cluster']

        if useTest:
            params = {'useTest': True,
                      'useScope': True,
                      'save': False,
                      'completeCV': False,
                      'dataAugm': False,
                      'drop_cols': ['scope', 'Date', 'real_target', 'pack', 'size (GM)', 'cluster'] + gte_cols,
                      'cluster': None,
                      'name': 'catboost'
                      }
        else:
            params = {'useTest': False,
                      'useScope': True,
                      'save': False,
                      'completeCV': False,
                      'dataAugm': True,
                      'drop_cols': ['scope', 'Date', 'real_target', 'pack', 'size (GM)', 'cluster'] + gte_cols,
                      'cluster': None,
                      'name': 'catboost'
                      }

        return model_params, params

