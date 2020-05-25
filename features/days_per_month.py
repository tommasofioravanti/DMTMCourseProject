from calendar import monthrange

def days_per_month(df_train):
    days_per_month = []

    for d in df_train.Date:
        d.to_pydatetime()
        days_per_month.append(monthrange(d.year, d.month)[1])

    df_train["days_per_month"] = days_per_month
    return df_train


if __name__ == '__main__':
    from preprocessing.preprocessing import convert_date
    import pandas as pd

    train = pd.read_csv("../dataset/original/train.csv")
    train = convert_date(train)
    train = days_per_month(train)

    print(train)