def lag_price(df, lag_shift):
    df = df.sort_values(['sku', 'Date'])
    df['lag_price_' + str(lag_shift)] = df[['Date', 'sku', 'price']].groupby('sku')['price'].shift(lag_shift).values
    return df


def lag_pos_exposed(df, lag_shift):
    df = df.sort_values(['sku', 'Date'])
    df['lag_POS_exposed_' + str(lag_shift)] = df[['Date', 'sku', 'POS_exposed w-1']].groupby('sku')['POS_exposed w-1']\
        .shift(lag_shift).values
    return df


def lag_volume_on_promo(df, lag_shift):
    df = df.sort_values(['sku', 'Date'])
    df['lag_volume_on_promo_' + str(lag_shift)] = df[['Date', 'sku', 'volume_on_promo w-1']].groupby('sku')[
        'volume_on_promo w-1'].shift(lag_shift).values
    return df


def lag_sales(df, lag_shift):
    df = df.sort_values(['sku', 'Date'])
    df['lag_sales_' + str(lag_shift)] = df[['Date', 'sku', 'sales w-1']].groupby('sku')['sales w-1']\
        .shift(lag_shift).values
    return df
