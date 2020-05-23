import pandas as pd
import numpy as np
from tqdm import tqdm

"""
WARNING: This feature returns the entire DataFrame, not the single columns
"""


# Moving Average
def moving_average(df, k, inverse_interpolation=True):
    # RETURN DF
    df = df.sort_values(['sku', 'Date']).reset_index(drop=True)
    curr_idx = 0
    previous_k_idx = curr_idx
    current_sku = None
    """
    k: indicates how much of the past consider [at most]
    """
    res = []
    for i, s, sal in tqdm(zip(df.index, df.sku, df['sales w-1'])):
        if not s == current_sku or current_sku is None:
            if inverse_interpolation:
                res.append(sal)
            else:
                res.append(np.nan)
            current_sku = s
            curr_idx += 1
            previous_k_idx = i
        else:
            avg = np.mean(df.loc[previous_k_idx:curr_idx, 'sales w-1'].values)

            res.append(avg)

            if curr_idx - previous_k_idx == k:
                previous_k_idx += 1
            curr_idx += 1

            if curr_idx - previous_k_idx == k + 1:
                print(f"Exceeded: {curr_idx - previous_k_idx}")

    df['moving_average'] = res
    return df