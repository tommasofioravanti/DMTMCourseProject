import pandas as pd
from datetime import datetime

# TODO:
# - feature che tiene conto delle vendite fatte complessivamente
# - feature che tiene conto delle vendite fatte negli ultimi 1,2,3,6 mesi
# - feature che tiene conto dell'incremento delle vendite da una settimana all'altra

def preprocessing_more(df_train: pd.DataFrame) -> pd.DataFrame:
    # first split the date into columns
    df_train = _split_date(df_train)
    # transform to categorical
    df_train = _to_categorical(df_train)
    df_train = df_train.fillna(0)
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

def convert_date(df):
    df['Unnamed: 0'] = df['Unnamed: 0'].str.split(" ")
    df['Unnamed: 0'] = df['Unnamed: 0'].apply(lambda x: "-".join(x[1:]))
    df = df.rename(columns={'Unnamed: 0':'Date'})
    df.Date = pd.to_datetime(df.Date)
    return df


def inverse_interpolation(df, date='2016-12-10'):
    first_we_idx = df[df.Date == pd.to_datetime(date)].index

    df = df.sort_values(['sku', 'Date'])
    for i in first_we_idx:
        inverse_interpolation = 2 * df.loc[i + 1, 'sales w-1'] - df.loc[i + 2, 'sales w-1']
        df.loc[i, 'sales w-1'] = inverse_interpolation

    return df