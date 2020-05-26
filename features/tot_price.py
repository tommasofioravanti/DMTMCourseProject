import numpy as np
from sklearn.linear_model import LinearRegression

def impute_last_salesW1(df_train,sku_list):
    #tot_price_wk1 = []

    output = []
    for i in sku_list:
        salesW1 = np.asarray(df_train[df_train.sku == i]['sales w-1'])
        sales1 = []
        for j in range(1, len(salesW1)):
            sales1.append(salesW1[j])
        sales1 = np.asarray(sales1).reshape(-1, 1)
        price = np.asarray(df_train[df_train.sku == i].price)

        price1 = (price[0:132]).reshape(-1, 1)
        target_X = price[132].reshape(-1, 1)
        # print(len(price),len(sales1))
        model = LinearRegression().fit(price1, sales1)
        # print(target_X)
        out = model.predict(target_X)
        output.append(out)
    return output


def tot_price_per_wk(df_train):
    #tot_price_wk1=[]

    sku_list = list(set(df_train.sku))
    sku_list.sort()
    output = impute_last_salesW1(df_train, sku_list)
    tot_sales = []
    n_price = []
    for i, sku in enumerate(sku_list):
        # print(i,sku_list)
        n_price.append(df_train[df_train.sku == sku].price)
        n_sales = np.asarray(df_train[df_train.sku == sku]['sales w-1'])[1:133]
        # print(output[i][0][0])
        tot_sales.append(np.append(n_sales, output[i][0][0]))
        # print(tot_sales)
    tot_sales = np.asarray(tot_sales)

    n_price = np.asarray(n_price)
    # print(tot_sales,n_price)
    tot_price_wk = tot_sales * n_price
    tot_price_new = []
    for i in tot_price_wk:
        for j in range(len(i)):
            tot_price_new.append(i[j])

    df_train['tot_cost'] = tot_price_new
    return df_train