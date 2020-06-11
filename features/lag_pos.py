import pandas as pd

def lag_pos(df, lag_shift):
    df = df.sort_values(['sku', 'Date'])
    df['lag_pos' + str(lag_shift)] = df[['Date', 'sku', 'POS_exposed w-1']].groupby('sku')['POS_exposed w-1'].shift(lag_shift).values
    return df