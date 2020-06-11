import pandas as pd

def lag_volume(df, lag_shift):
    df = df.sort_values(['sku', 'Date'])
    df['lag_volume_' + str(lag_shift)] = df[['Date', 'sku', 'volume_on_promo w-1']].groupby('sku')['volume_on_promo w-1'].shift(lag_shift).values
    return df