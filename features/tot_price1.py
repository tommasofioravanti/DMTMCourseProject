import numpy as np


def tot_price_per_wk(df):

    sku_list = list(set(df.sku))
    sku_list.sort()
    t_p = []
    for i in sku_list:

        price_1 = np.asarray(df[df.sku == i]['price'].shift(1))
        salesw1=np.asarray(df[df.sku==i]['sales w-1'])
        t_price=price_1*salesw1
        t_p.append(t_price)




    #print(np.concatenate(tot_price_wk, axis=0))
    tot_price_wk = np.concatenate(t_p, axis=0)


    df['tot_cost'] = tot_price_wk
    return df
