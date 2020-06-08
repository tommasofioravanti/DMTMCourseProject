import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

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


def train_validation_split(train, k=0.20, same_months_test=False):
    train = train.sort_values('Date')

    if same_months_test:
        # Si cerca di considerare gli stessi mesi del test
        start_date = pd.to_datetime('2017-06-29')
        end_date = pd.to_datetime('2017-12-15')
        mask_val = (train.Date >= start_date) & (train.Date <= end_date)
        val = train[mask_val]
        train = train[train.Date < start_date]
        val_dates = val.sort_values('Date').drop_duplicates('Date')['Date']

    else:
        train_dates = train['Date'].drop_duplicates(keep='first')
        k = int(len(train_dates) * k)
        val_dates = train_dates[-k:]
        train, val = train[~train.Date.isin(val_dates)], train[train.Date.isin(val_dates)]

    return train, val, val_dates


def data_augmentation(df):
    df_2017 = df[df.Date.dt.year == 2017]
    df_2018 = df[df.Date.dt.year == 2018]
    df_2017['order'] = df_2017.sort_values(['sku', 'Date']).groupby('sku').cumcount()
    df_2018['order'] = df_2018.sort_values(['sku', 'Date']).groupby('sku').cumcount()
    res = []
    for d in df[df.Date.dt.year == 2016].Date.drop_duplicates().values:
        res.append(d)
    start_date = pd.to_datetime('2016-12-10')
    while len(res) != 52:
        date = start_date - pd.to_timedelta(7, unit='d')
        res.append(date)
        start_date = date
    res = pd.to_datetime(res)
    res = res.sort_values()
    res = [res] * 43
    res = [y for x in res for y in x]
    df_2016 = df_2017.copy()
    df_2016 = df_2016.sort_values(['sku', 'Date'])
    df_2016['Date'] = res
    df_2016['order'] = df_2016.sort_values(['sku', 'Date']).groupby('sku').cumcount()

    cols = ['price', 'POS_exposed w-1', 'volume_on_promo w-1', 'sales w-1', 'target']
    for c in cols:
        df_2016[c] = (df_2016[c].values + df_2018[c].values) / 2

    df_dates = df[df.Date.dt.year == 2016].Date.drop_duplicates().values
    df_2016 = df_2016.drop('order', axis=1)
    df_2016 = df_2016.sort_values(['sku', 'Date'])
    df = df.sort_values(['sku', 'Date'])

    df_dates = df_dates[1:]

    cols = ['price', 'POS_exposed w-1', 'volume_on_promo w-1', 'sales w-1', 'target']
    df_2016.loc[df_2016.Date.isin(df_dates), cols] = df[df.Date.isin(df_dates)][cols].values

    cols = ['price', 'target']
    df_2016.loc[df_2016.Date == '2016-12-10', cols] = df[df.Date == '2016-12-10'][cols].values

    df_2016_index = df[df.Date.dt.year == 2016].index
    df = df.drop(df_2016_index)

    df = pd.concat([df_2016, df])
    df = df.sort_values(['sku', 'Date']).reset_index(drop=True)
    df['sales w-1'] = df.groupby('sku').target.shift(1)

    return df


def preprocessing(train, test, useTest=True, dataAugmentation=False):

    if dataAugmentation:
        train = data_augmentation(convert_date(train))
        test = convert_date(test)
        df = pd.concat([train, test])
    else:
        df = pd.concat([train, test])
        df = convert_date(df)

    if useTest:
        # TODO Riga da RIMUOVERE PRIMA DELLA CONSEGNA
        df.loc[df.target.isna(), 'target'] = df[df.target.isna()][['Date', 'sku', 'sales w-1']].groupby('sku')['sales w-1'].shift(-1).values

    df = df.sort_values(['sku', 'Date']).reset_index(drop=True)
    # Encoding Categorical Features
    le = LabelEncoder()
    df.pack = le.fit_transform(df.pack)
    df.brand = le.fit_transform(df.brand)

    # Impute sales w-1 NaNs in the first week
    first_date = df.Date.sort_values().drop_duplicates().values
    first_date = first_date[0]
    df = inverse_interpolation(df, date=first_date)

    #   --------------- Features -----------------

    """The log function essentially de-emphasizes very large values.
    It is more easier for the model to predict correctly if the distribution is not that right-skewed which is
    corrected by modelling log(sales) than sales."""

    # real_values = df[['Date', 'sku', 'target']].rename(columns={'target':'real_target'})
    df['real_target'] = df.target
    df['target'] = np.log1p(df.target.values)
    df['sales w-1'] = np.log1p(df['sales w-1'].values)

    return df