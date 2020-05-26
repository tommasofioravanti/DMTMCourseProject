import pandas as pd

def partial_sales(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    For each sku tells the total number of sells up to that week
    """
    dic_sku_sales = dict(zip(list(set(df_train.sku)), [0 for _ in range(len(list(set(df_train.sku))))]))
    new_col = []
    for sku,sales in zip(df_train.sku, df_train['sales w-1']):
        dic_sku_sales[sku] += sales
        new_col.append(dic_sku_sales[sku])
    df_train['partial_sales'] = new_col
    return df_train