import pandas as pd
import numpy as np
from tqdm import tqdm

def exponential_weighted_moving_average(df, com=0.5):
    df = df.sort_values(['sku','Date'])
    sku_values = list(set(df.sku.values))
    sku_values = sorted(sku_values)
    res = []
    for s in tqdm(sku_values):
        ewma = df[df.sku==s]['sales w-1'].ewm(com=com).mean().values
        res.append(ewma)
    res = [item for x in res for item in x]
    return res