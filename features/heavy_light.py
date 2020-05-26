import pandas as pd

def heavy_light(df):
    """
    Simply divides into heavy and light objects (the threshold is 350g)
    TODO: find a better threshold, maybe more categories?
    """
    new_col = []
    for gms in df['size (GM)']:
        if gms < 350:
            new_col.append(0)
        else:
            new_col.append(1)
    df['heavy_light'] = new_col
    return df