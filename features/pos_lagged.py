import pandas as pd

def pos_lagged(df):
    df['pos_lagged_2'] = df['POS_exposed w-1'].shift(periods=1)
    return df