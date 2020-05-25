import pandas as pd
import numpy as np

def lag_target(df, lag_shift):
    df = df.sort_values(['sku', 'Date'])

    df['lag_target_' + str(lag_shift)] = df[['Date', 'sku', 'target']].groupby('sku')['target'].shift(lag_shift).values
    return df