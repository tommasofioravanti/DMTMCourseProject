import numpy as np
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

    n_price = np.asarray(n_price)

    tot_price_wk = tot_sales * n_price

    #print(np.concatenate(tot_price_wk, axis=0))
    tot_price_wk = np.concatenate(tot_price_wk, axis=0)


    df['tot_cost'] = tot_price_wk
    return df
