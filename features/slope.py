import pandas as pd
import numpy as np
from tqdm import tqdm

def slope(df):
    """
    slope values = [0,+1, -1], res_changes=[sales w-1 - sales w-2]
    """
    df = df.sort_values(['sku', 'Date']).reset_index(drop=True)
    curr_idx = 0
    current_sku = None
    res_slope = []
    res_changes = []
    for i, s, s_1 in tqdm(zip(df.index, df.sku, df['sales w-1'])):
        if not current_sku == s or current_sku is None:
            current_sku = s
            curr_idx += 1
            res_slope.append(0)
            res_changes.append(0)

        else:
            res_slope.append(np.sign(s_1 - df.loc[curr_idx - 1, 'sales w-1']))
            res_changes.append(s_1 - df.loc[curr_idx - 1, 'sales w-1'])
            curr_idx += 1
    return res_slope, res_changes