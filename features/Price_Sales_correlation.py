import numpy as np
from scipy.stats import pearsonr
from scipy.stats import zscore
from sklearn import preprocessing

from sklearn.linear_model import LinearRegression

def impute_last_salesW1(df,sku_list):

    output = []

    for i in sku_list:
        salesW1 = np.asarray(df[df.sku == i]['sales w-1'])
        sales1 = []
        # per ogni sku si prendono i salesW1 tranne il primo valore (che è Nan)
        for j in range(1, len(salesW1)):
            sales1.append(salesW1[j])
        sales1 = np.asarray(sales1).reshape(-1, 1)

        price = np.asarray(df[df.sku == i].price)

        price1 = (price[0:(price.shape[0] - 1)]).reshape(-1, 1)
        target_X = price[(price.shape[0] - 1)].reshape(-1, 1)
        # print(len(price1),len(sales1))

        # si usa una Linear regression con valori di input price1 e target sales1 (per il train) per
        # imputare l'ultimo valore di salesW1 per ogni sku
        model = LinearRegression().fit(price1, sales1)
        # print(target_X)
        out = model.predict(target_X)
        output.append(out)
    return output


def tot_price_per_wk(df):

    sku_list = list(set(df.sku))
    sku_list.sort()
    output = impute_last_salesW1(df, sku_list)
    tot_sales = []
    n_price = []
    for i, sku in enumerate(sku_list):

        n_price.append(df[df.sku == sku].price)
        n_sales = np.asarray(df[df.sku == sku]['sales w-1'])
        n_sales = n_sales[1:(n_sales.shape[0])]
        tot_sales.append(np.append(n_sales, output[i][0][0]))

    # tot_sales e n_price adesso hanno la stessa lunghezza e corrispondono alla stessa settimana
    tot_sales = np.asarray(tot_sales)

    return tot_sales
def conc_corr(df):
    tot_sales=tot_price_per_wk(df)
    tot_sales = np.asarray(tot_sales)
    sku_list = list(set(df.sku))
    sku_list.sort()

    pear = {}
    #p=[144,546]
    for j, i in enumerate(sku_list):
        df1 = df
        p1=[]
        for e in range(1, (df1[df1.sku == i].shape[0])):
            a = preprocessing.scale((df1[df1.sku == i].iloc[0:e + 1]).price)
            # print(a)
            # v=len(tot_sales[j])
            b = preprocessing.scale(tot_sales[j][0:e + 1])
            pearson = pearsonr(a, b)[0]

            # print(p1)
            if (np.isnan(pearson).any()):
                pearson = 0
            p1.append(pearson)
            # print(p)
        pear.update({i: p1})
    #print(len(pear))
    for j, i in enumerate(sku_list):
        tot_sal = tot_sales[j]
        sales1 = []
        for k in range(1, len(tot_sal)):
            sales1.append(tot_sal[k])
        sales1 = np.asarray(sales1).reshape(-1, 1)
        price = np.asarray(df[df.sku == i].price)

        price1 = (price[1:(price.shape[0])]).reshape(-1, 1)
        # print(X)
        X = np.vstack((price1, sales1)).reshape(-1, 2)
        Y_train = pear.get(i)
        # print(Y_train,len(Y_train))
        target_price = price[0].reshape(-1, 1)
        target_sales = tot_sal[0].reshape(-1, 1)
        target_X = np.vstack((target_price, target_sales)).reshape(-1, 2)
        # print(len(price),len(sales1))
        # print(X)
        model = LinearRegression().fit(X, Y_train)
        # print(target_X)

        out = model.predict(target_X)
        (pear.get(i)).insert(0, float(out))

    lista = []
    for i in sku_list:
        lista.append(pear.get(i))
    flat_list = []
    for sublist in lista:
        for item in sublist:
            flat_list.append(item)

    df['Corr'] = flat_list
    return df


