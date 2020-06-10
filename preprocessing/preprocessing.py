import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

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
        imputed_sales = 2 * df.loc[i + 1, 'sales w-1'] - df.loc[i + 2, 'sales w-1']
        df.loc[i, 'sales w-1'] = imputed_sales

        # Se è la prima riga del train originale imputo anche POS_exp e volume
        # Se è la prima riga della data_augmentation imputo solo sales
        if date == '2016-12-10':
            imputed_pos_exp = 2 * df.loc[i + 1, 'POS_exposed w-1'] - df.loc[i + 2, 'POS_exposed w-1']
            df.loc[i, 'POS_exposed w-1'] = imputed_pos_exp
            imputed_volume = 2 * df.loc[i + 1, 'volume_on_promo w-1'] - df.loc[i + 2, 'volume_on_promo w-1']
            df.loc[i, 'volume_on_promo w-1'] = imputed_volume

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


def data_augmentation_2(df, random_noise=False):
    df17 = df[df.Date.dt.year == 2017].sort_values(['sku', 'Date']).reset_index(drop=True)
    df18 = df[df.Date.dt.year == 2018].sort_values(['sku', 'Date']).reset_index(drop=True)

    df16 = df17.merge(df18, how='left', on=['sku', 'pack', 'size (GM)', 'brand', 'scope'], left_index=True,
                      right_index=True)
    df16 = df16.sort_values(['sku', 'Date_x'])

    np.random.seed(42)
    weight_17 = np.random.uniform(0, 1, df16.shape[0])
    weight_18 = 1 - weight_17

    cols = ['price', 'POS_exposed w-1', 'volume_on_promo w-1', 'sales w-1', 'target']
    for c in cols:
        df16[c] = (weight_17 * df16[c + '_x'].values) + (weight_18 * df16[c + '_y'].values)

    drop_cols = [c for c in df16.columns if '_x' in c or '_y' in c]
    df16 = df16.drop(drop_cols, axis=1)
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
    df16['Date'] = res

    if random_noise:
        # Random Perturbation
        def random_perturbation(df, col):
            np.random.seed(42)
            stats = df.groupby('sku')[col].agg(mean=np.mean, var=np.var, n='count', min_='min', max_='max').reset_index()
            df_tmp = df.merge(stats, how='left', on='sku')
            df_tmp = df_tmp.fillna(0)
            values = []
            for v, mean in zip(df_tmp[col], df_tmp['mean']):
                noise = np.random.normal(0, 1, 1)[0]
                new_v = v + (noise * (mean / 4))
                values.append(new_v)
            df_tmp['new_' + col] = values
            negative_indices = np.where(df_tmp['new_' + col] < 0)[0]
            print(f'{c} -- > Negative values {negative_indices.shape}')
            if negative_indices.shape[0] > 0:
                df_tmp.loc[negative_indices, 'new_' + col] = 0.0
            return df_tmp['new_' + col].values

        cols = ['price', 'POS_exposed w-1', 'volume_on_promo w-1', 'sales w-1', 'target']
        for c in cols:
            df16[c] = random_perturbation(df16, c)

    df16 = df16.drop(df16[df16.Date >= '2016-12-10'].index)
    df = pd.concat([df16, df])
    df['sales w-1'] = df.groupby('sku').target.shift(1)

    return df


def preprocessing(train, test, useTest=True, dataAugmentation=False,rand_noise=False):

    if dataAugmentation:
        train = data_augmentation_2(convert_date(train), random_noise=rand_noise)
        test = convert_date(test)
        df = pd.concat([train, test])

        df = df.sort_values(['sku', 'Date']).reset_index(drop=True)
        nan_indices = df[df['POS_exposed w-1'].isna()].index

        for i in nan_indices:
            imputed_pos = (df.loc[i - 1, 'POS_exposed w-1'] + df.loc[i + 1, 'POS_exposed w-1']) / 2
            df.loc[i, 'POS_exposed w-1'] = imputed_pos
            imputed_volume = (df.loc[i - 1, 'volume_on_promo w-1'] + df.loc[i + 1, 'volume_on_promo w-1']) / 2
            df.loc[i, 'volume_on_promo w-1'] = imputed_volume
    else:
        df = pd.concat([train, test])
        df = convert_date(df)

    if useTest:
        # TODO Riga da RIMUOVERE PRIMA DELLA CONSEGNA    # In realtà credo vada bene questa riga, l'importante è non usare sales w-1 delle settimane future
        df.loc[df.target.isna(), 'target'] = df[df.target.isna()][['Date', 'sku', 'sales w-1']].groupby('sku')['sales w-1'].shift(-1).values

    df = df.sort_values(['sku', 'Date']).reset_index(drop=True)
    # Encoding Categorical Features
    le = LabelEncoder()
    df.pack = le.fit_transform(df.pack)
    df.brand = le.fit_transform(df.brand)

    # Impute NaNs in the first week if not data_augmentation

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


def ohe_categorical(df, categorical_features):
    for c in categorical_features:
        dummy = pd.get_dummies(df[c], prefix=c)
        df[dummy.columns] = dummy
    return df