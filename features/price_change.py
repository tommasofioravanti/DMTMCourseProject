import pandas
def price_change(df):
    df = df.sort_values(['sku', 'Date'])
    df['price_change'] = df[['Date', 'sku', 'price']].groupby('sku')['price'].shift(1).values - df['price']
    return df