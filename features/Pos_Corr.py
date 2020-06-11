import numpy as np
from preprocessing.preprocessing import inverse_interpolation
from scipy.stats import pearsonr

def impute_price(df):
    df=inverse_interpolation(df)
    return df



def Corr(df):

    sku_list = list(set(df.sku))
    sku_list.sort()

    df=impute_price(df)

    #t_p = []


    pear = {}
    for j, i in enumerate(sku_list):
        df1 = df
        p1 = []
        for e in range(1, (df1[df1.sku == i].shape[0])):
            a = df1[df1.sku == i].iloc[0:e + 1]['POS_exposed w-1']

            # print(a)
            # v=len(tot_sales[j])
            b = df1[df1.sku==i].iloc[0:e+1]['sales w-1']
            pearson = pearsonr(a, b)[0]

            # print(p1)
            if (np.isnan(pearson).any()):
                pearson = 0
            p1.append(pearson)
            # print(p)
        pear.update({i: p1})
        # print(len(pear))
    lista = []
    for i in sku_list:
        pear.get(i).insert(0,0)
    for i in sku_list:
        lista.append(pear.get(i))
    flat_list = []
    for sublist in lista:
        for item in sublist:
            flat_list.append(item)

    df['Corr'] = flat_list
    return df


