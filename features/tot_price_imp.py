import numpy as np
from preprocessing.preprocessing import inverse_interpolation_price


def impute_price(df):
    df=inverse_interpolation_price(df)
    return df



def tot_price_per_wk(df):

    sku_list = list(set(df.sku))
    sku_list.sort()
    df['price']=df.groupby('sku')['price'].shift(1)
    df=impute_price(df)

    t_p = []
    for i in sku_list:

        price_1 = np.asarray(df[df.sku==i]['price'])

        salesw1=np.asarray(df[df.sku==i]['sales w-1'])
        t_price=price_1*salesw1
        t_p.append(t_price)




    #print(np.concatenate(tot_price_wk, axis=0))
    tot_price_wk = np.concatenate(t_p, axis=0)


    df['tot_cost_imp'] = tot_price_wk
    return df
