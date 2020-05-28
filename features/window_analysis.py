import pandas as pd


def mean_rolling_window(df, attr_name: str, winsize=2, extended=True) -> pd.DataFrame:
    """
    For each row (and for each sku) compute the MEAN of n previous rows (extended -> all previous rows)
    @param df:
    @param attr_name: name of the column to take into account
    @param winsize: with extended keep it always 2
    @param extended: non fixed window mode
    @return: df ordered by sku
    """
    df = df.sort_values(['sku', 'Date'])

    # Default branch
    if extended:
        extended_window = []

        for s in sorted(set(df.sku)):
            window_single_sku = df[df.sku == s][attr_name].expanding(winsize).mean()
            extended_window.extend(window_single_sku)

        df["mean_" + str(attr_name) + "_extended"] = extended_window

    else:
        rolling_window = []

        for s in sorted(set(df.sku)):
            window_single_sku = df[df.sku == s][attr_name].rolling(window=winsize).mean()
            rolling_window.extend(window_single_sku)

        df["mean_" + str(attr_name) + "_rolled"] = rolling_window

    return df


def max_rolling_window(df, attr_name: str, winsize=2, extended=True) -> pd.DataFrame:
    """
    For each row (and for each sku) compute the MAX of n previous rows (extended -> all previous rows)
    @param df:
    @param attr_name: name of the column to take into account
    @param winsize: with extended keep it always 2
    @param extended: non fixed window mode
    @return: df ordered by sku
    """
    df = df.sort_values(['sku', 'Date'])

    # Default branch
    if extended:
        extended_window = []

        for s in sorted(set(df.sku)):
            window_single_sku = df[df.sku == s][attr_name].expanding(winsize).max()
            extended_window.extend(window_single_sku)

        df["max_" + str(attr_name) + "_extended"] = extended_window

    else:
        rolling_window = []
        df = df.sort_values(['sku', 'Date'])

        for s in sorted(set(df.sku)):
            window_single_sku = df[df.sku == s][attr_name].rolling(window=winsize).max()
            rolling_window.extend(window_single_sku)

        df["max_" + str(attr_name) + "_rolled"] = rolling_window

    return df


def min_rolling_window(df, attr_name: str, winsize=2, extended=True) -> pd.DataFrame:
    """
    For each row (and for each sku) compute the MIN of n previous rows (extended -> all previous rows).
    @param df:
    @param attr_name: name of the column to take into account
    @param winsize: with extended keep it always 2
    @param extended: non fixed window mode
    @return: df ordered by sku
    """
    df = df.sort_values(['sku', 'Date'])

    # Default branch
    if extended:
        extended_window = []

        for s in sorted(set(df.sku)):
            window_single_sku = df[df.sku == s][attr_name].expanding(winsize).min()
            extended_window.extend(window_single_sku)

        df["min_" + str(attr_name) + "_extended"] = extended_window

    else:
        rolling_window = []
        df = df.sort_values(['sku', 'Date'])

        for s in sorted(set(df.sku)):
            window_single_sku = df[df.sku == s][attr_name].rolling(window=winsize).min()
            rolling_window.extend(window_single_sku)

        df["min_" + str(attr_name) + "_rolled"] = rolling_window

    return df


if __name__ == '__main__':
    import pandas as pd
    from preprocessing.preprocessing import convert_date, inverse_interpolation

    train = pd.read_csv("../dataset/original/train.csv")
    train = convert_date(train)
    train = inverse_interpolation(train)

    train = mean_rolling_window(train, "sales w-1", extended=False, winsize=12)
    train = max_rolling_window(train, "sales w-1", extended=False, winsize=12)
    train = min_rolling_window(train, "sales w-1", extended=False, winsize=12)

    print(train[train.sku==144].to_string())
