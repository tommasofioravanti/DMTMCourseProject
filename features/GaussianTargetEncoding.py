import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os

import sys
sys.path.append('../')

from preprocessing.preprocessing import preprocessing
from features.clustering import get_cluster

class GaussianTargetEncoder(object):

    def __init__(self, group_cols, target_col='target', prior_cols=None):
        self.target_col = target_col
        self.prior_cols = prior_cols

        if isinstance(group_cols, str):
            self.group_cols = [group_cols]
        else:
            self.group_cols = list(group_cols)

    #  Nel nostro caso gli passiamo il train e tiriamo fuori gli Encoding per la week successiva

    def get_prior(self, df):
        if self.prior_cols is None:
            prior = np.full(len(df), df[self.target_col].mean())
        else:
            prior = df[self.prior_cols].mean(axis=1)
        return prior

    # Fittiamo sul train
    def fit(self, df, window):
        self.stats = df.assign(mu_prior=self.get_prior(df), y=df[self.target_col])
        # if len(self.dates) < window:
        #     self.stats = self.stats.groupby(self.group_cols).agg(
        #         n=('y', 'count'),
        #         mu_mle=('y', np.mean),
        #         sig2_mle=('y', np.var),
        #         mu_prior=('mu_prior', np.mean)
        #     )
        #
        # else:
        #self.dates = self.dates[-window:]
        self.stats = self.stats.groupby(self.group_cols).agg(
            n=('y', 'count'),
            mu_mle=('y', np.mean),
            sig2_mle=('y', np.var),
            mu_prior=('mu_prior', np.mean)
        )


    # Trasformiamo il validation
    def transform(self, df, prior_precision=1000, stat_type='mean'):

        precision = prior_precision + self.stats.n / self.stats.sig2_mle

        if stat_type == 'mean':
            numer = prior_precision * self.stats.mu_prior + (self.stats.n / self.stats.sig2_mle) * self.stats.mu_mle

            denom = precision

        elif stat_type == 'var':
            numer = 1.0
            denom = precision

        elif stat_type == 'precision':
            numer = precision
            denom = 1.0
        else:
            raise ValueError(f"stat_type={stat_type} not recognized.")

        mapper = dict(zip(self.stats.index, numer / denom))

        if isinstance(self.group_cols, str):
            keys = df[self.group_cols].values.tolist()
        elif len(self.group_cols) == 1:
            keys = df[self.group_cols[0]].values.tolist()
        else:
            keys = zip(*[df[x] for x in self.group_cols])

        # keys = [tuple(x) for x in keys]
        # print(f'mapper keys: {mapper.keys()} \n keys: {keys}')
        res = []
        count = 0
        not_in_mapper = []
        for k in keys:
            if k in mapper:
                res.append(mapper[k])
            else:
                not_in_mapper.append(k)
                res.append(np.nan)
                count += 1
        if count >1:
            #print(f'Groupby:{self.group_cols} --> {set(not_in_mapper)} not in mapper: {mapper.keys()}')
            #print(f'Number of nans: {count}')
            pass

        values = np.array(res).astype(float)
        #values = np.array([mapper[k] for k in keys]).astype(float)

        prior = self.get_prior(df)
        values[~np.isfinite(values)] = prior[~np.isfinite(values)]
        return values

    def fit_transform(self, df, prior_precision=1000, window=2):
        self.fit(df, window)
        return self.transform(df, prior_precision)


def run_gte_feature():
    abs_path = Path(__file__).absolute().parent
    train_path = os.path.join(abs_path, "../dataset/original/train.csv")
    test_path = os.path.join(abs_path, "../dataset/original/x_test.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    df = preprocessing(train, test, useTest=True, dataAugmentation=True)
    # df, categorical_f = add_all_features(df)
    df_cluster = get_cluster()
    df = df.merge(df_cluster, how='left', on='sku')

    def simple_gen(df):
        df = df.sort_values('Date')
        dates = df[df.Date >= '2016-12-10']['Date'].drop_duplicates().values
        dates = dates[1:]
        for d in dates:
            yield df[df.Date < d], df[df.Date == d]

    gen = simple_gen(df)

    group_and_priors = {
        ('pack'): None,
        ('brand'): None,
        ('cluster'): None,
        ('pack', 'brand'): ['gte_pack', 'gte_brand'],
        ('pack', 'cluster'): ['gte_pack', 'gte_cluster'],
        ('brand', 'cluster'): ['gte_brand', 'gte_cluster'],
        ('pack', 'brand', 'cluster'): ['gte_pack_brand', 'gte_pack_cluster'],
    }

    df_gte = pd.DataFrame()

    window = 8
    prior_precision = 50
    for t, v in tqdm(gen):
        date = v.Date.drop_duplicates(keep='first')
        features = []
        for group_cols, prior_cols in group_and_priors.items():
            if isinstance(group_cols, str):
                f_name = "gte_" + group_cols
            else:
                f_name = "gte_" + '_'.join(group_cols)
            features.append(f_name)
            gte = GaussianTargetEncoder(group_cols, 'target', prior_cols)

            dates = t.Date.drop_duplicates()
            if len(dates) > window:
                t = t[t.Date.isin(dates[-window:])]

            #print(f'Encoding Train: days < {date} : rows {t.shape[0]}')
            t.loc[:, features[-1]] = gte.fit_transform(t, prior_precision=prior_precision, window=window)

            #print(f'Encoding Validation = {date} \n')
            v.loc[:, features[-1]] = gte.transform(v, prior_precision=prior_precision)
        df_gte = pd.concat([df_gte, v])

    gte_cols = [x for x in df_gte.columns if 'gte' in x]

    save_path = os.path.join(abs_path, f"gte_features_w{window}_prp{prior_precision}.csv")
    df_gte[['Date', 'sku', 'target', 'real_target'] + gte_cols].to_csv(save_path, index=False)


if __name__=='__main__':
    train = pd.read_csv("../dataset/original/train.csv")
    test = pd.read_csv("../dataset/original/x_test.csv")

    df = preprocessing(train, test, useTest=True, dataAugmentation=True)
    #df, categorical_f = add_all_features(df)

    def simple_gen(df):
        df = df.sort_values('Date')
        dates = df[df.Date >= '2016-12-10']['Date'].drop_duplicates().values
        dates = dates[1:]
        for d in dates:
            yield df[df.Date < d], df[df.Date == d]

    gen = simple_gen(df)

    group_and_priors = {
        ('pack'): None,
        ('brand'): None,
        ('cluster'): None,
        ('pack', 'brand'): ['gte_pack', 'gte_brand'],
        ('pack', 'cluster'): ['gte_pack', 'gte_cluster'],
        ('brand', 'cluster'): ['gte_brand', 'gte_cluster'],
        ('pack', 'brand', 'cluster'): ['gte_pack_brand', 'gte_pack_cluster'],
    }

    df_gte = pd.DataFrame()

    window = 8
    prior_precision = 50
    for t, v in tqdm(gen):
        date = v.Date.drop_duplicates(keep='first')
        features = []
        for group_cols, prior_cols in group_and_priors.items():
            if isinstance(group_cols, str):
                f_name = "gte_" + group_cols
            else:
                f_name = "gte_" + '_'.join(group_cols)
            features.append(f_name)
            gte = GaussianTargetEncoder(group_cols, 'target', prior_cols)

            dates = t.Date.drop_duplicates()
            if len(dates) > window:
                t = t[t.Date.isin(dates[-window:])]

            print(f'Encoding Train: days < {date} : rows {t.shape[0]}')
            t.loc[:,features[-1]] = gte.fit_transform(t, prior_precision=prior_precision, window=window)


            print(f'Encoding Validation = {date} \n')
            v.loc[:,features[-1]] = gte.transform(v, prior_precision=prior_precision)
        df_gte = pd.concat([df_gte, v])

    gte_cols = [x for x in df_gte.columns if 'gte' in x]
    df_gte[['Date', 'sku', 'target', 'real_target'] + gte_cols].to_csv(f"gte_features_w{window}_prp{prior_precision}.csv", index=False)

