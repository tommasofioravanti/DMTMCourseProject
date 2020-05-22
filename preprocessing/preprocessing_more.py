import pandas as pd
from datetime import datetime

def preprocessing_more(df_train: pd.DataFrame) -> pd.DataFrame:
    # first split the date into columns
    df_train = _split_date(df_train)
    # transform to categorical
    df_train = _to_categorical(df_train)
    df_train = df_train.fillna(-1)
    df_train = df_train.sort_values(by=['year','month','day']).reset_index(drop=True)
    return df_train
    

def _split_date(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Split date into days, month and year cols
    """
    date_list = [[x.split(" ")[1],x.split(" ")[2], x.split(" ")[3]] for x in df_train["Unnamed: 0"]]
    df_train['day'] = [int(x[0]) for x in date_list]
    df_train['month'] = [int(datetime.strptime(x[1], "%B").month) for x in date_list]
    df_train['year'] = [int(x[2]) for x in date_list]
    return df_train.drop(['Unnamed: 0'], axis=1)

def _to_categorical(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Transform categorical to code
    """
    # brand
    df_train["brand"] = df_train["brand"].astype('category')
    df_train["brand"] = df_train["brand"].cat.codes
    df_train["pack"] = df_train["pack"].astype('category')
    df_train["pack"] = df_train["pack"].cat.codes
    return df_train