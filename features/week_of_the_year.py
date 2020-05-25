from datetime import datetime

# Works with convert_date(df)
def week_of_the_year(df_train):
    week_of_the_year = []

    for d in df_train.Date:
        week_of_the_year.append(datetime.date(d).isocalendar()[1])

    df_train["week_of_the_year"] = week_of_the_year
    return df_train
