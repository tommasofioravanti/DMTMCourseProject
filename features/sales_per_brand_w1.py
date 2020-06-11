import pandas as pd
def sales_per_brand_w1(df):
    grouped = pd.DataFrame({'sales_per_brand_w1' : df.groupby(['Date','brand'])['sales w-1'].sum()}).reset_index()
    return pd.merge(df, grouped, how='left')