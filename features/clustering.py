import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy import stats
from scipy.stats import pearsonr

import sys
sys.path.append('../')

from preprocessing.preprocessing import convert_date


def get_cluster():
    train = pd.read_csv("../dataset/original/train.csv")
    test = pd.read_csv("../dataset/original/x_test.csv")
    df = pd.concat([train, test], sort=False)

    df = convert_date(df)
    df = df.sort_values(['sku', 'Date'])
    # min(df[df.sku==2718].Date)

    series1 = []
    for i, s in enumerate(set(train.sku)):
        series1.append(((train[train.sku == s].target).values))

    series1 = np.asarray(series1)

    z1 = linkage(series1, 'single', metric='correlation')

    label_corr = fcluster(z1, 4, criterion='maxclust')  # Cluster with "correlation distance measure"
    sku = list((set(train.sku)))

    data = {'cluster': label_corr,
            'sku': sku}
    df_cluster = pd.DataFrame(data, columns=['cluster', 'sku'])

    return df_cluster