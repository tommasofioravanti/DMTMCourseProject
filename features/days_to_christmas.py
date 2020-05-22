from datetime import datetime, date, timedelta

def days_to_christmas(df_train):
    days_to_christmas = []
    for d,m,y in zip(df_train.day, df_train.month, df_train.year):
        days_to_christmas.append((date(y,12,25)-date(y,m,d)).days)
    df_train['days_to_christmas'] = days_to_christmas
    return df_train