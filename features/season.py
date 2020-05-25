# Works with convert_date(df)

def season(df_train):
    seasons = []

    for d in df_train.Date:
        d.to_pydatetime()
        seasons.append((d.month % 12 + 3)//3)

    df_train["seasons"] = seasons
    return df_train

