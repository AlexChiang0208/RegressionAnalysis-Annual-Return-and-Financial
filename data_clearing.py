### 預測性資料 ###

import pandas as pd
import numpy as np

path = '/Users/alex_chiang/Documents/迴歸分析/期末_多元迴歸/'

stock = pd.read_csv(path + '原始資料/' + 'stock_price.csv', encoding = 'cp950', 
                    parse_dates=True, index_col='年月日')

industry = pd.read_csv(path + '原始資料/' + 'industry.csv', 
                       encoding = 'cp950', index_col='年月')

financial_index = pd.read_csv(path + '原始資料/' + 'untidy_features.csv', 
                              encoding = 'cp950', index_col='年月')

dt1 = [str(industry.index[i])+'01' for i in range(len(industry.index))]
industry.index = pd.to_datetime(dt1)
industry.rename_axis('年月', inplace = True)

dt2 = [str(financial_index.index[i])+'01' for i in range(len(financial_index.index))]
financial_index.index = pd.to_datetime(dt2)
financial_index.rename_axis('年月', inplace = True)


# 整理 Y
stock.set_index(['證券代碼'], append = True, inplace = True)
stock = stock.swaplevel(axis = 0)
stock.sort_index(level = 0, inplace = True)

stock.drop(columns = ['簡稱'], inplace = True)
stock.rename(columns = {'收盤價(元)':'price'}, inplace = True)
stock.rename_axis(['stock_id', 'Time'], inplace = True)


stock_price = stock.unstack(0)['price']
stock_price = stock_price.iloc[5:]

Y_ExpRet = stock_price.loc[:'2018-03-31']
Y_ExpRet.dropna(axis = 1, inplace = True)


Y_ExpRet = Y_ExpRet.pct_change()
Y_ExpRet = Y_ExpRet.shift(-1)
Y_ExpRet = Y_ExpRet.iloc[range(0,17,2)]


idx = Y_ExpRet.columns
stock_price = stock_price.loc['2018-04-02':'2021-03-31'][idx]
stock_price = stock_price.dropna(axis=1)

idx = stock_price.columns
Y_ExpRet = Y_ExpRet[idx]

Y_ExpRet.isnull().values.any()
stock_price.isnull().values.any()


# 整理 X
X_feature = financial_index.merge(industry, on = ['證券代碼', '簡稱', '年月'])
X_feature.set_index(['證券代碼'], append = True, inplace = True)
X_feature = X_feature.swaplevel(axis = 0)
X_feature.sort_index(level = 0, inplace = True)


X_feature.drop(columns = ['簡稱', '交易所子產業代碼', 
                          'TEJ主產業代碼', 'TEJ子產業代碼'], inplace = True)
X_feature.rename_axis(['stock_id', 'Time'], inplace = True)
X_feature.iloc[:, :-1] = X_feature.iloc[:, :-1].applymap(lambda x: np.nan if x == 'N.A.' else float(x))


idx_new  = [i for i in X_feature.xs('2009-03-01', level = 1).index if i in idx]
X_feature = X_feature.loc[idx_new]

for i in idx_new:
    if len(X_feature.loc[i]) != 13 or X_feature.loc[i].isnull().values.any() != False:
        X_feature.drop(index = i, axis = 0, level = 0, inplace = True)

for i in X_feature.xs('2009-03-01', level = 1).index:
    X_feature.loc[i, '交易所主產業代碼'].replace('N.A.', method = 'ffill', inplace = True)
    X_feature.loc[i, '交易所主產業代碼'].replace('N.A.', method = 'bfill', inplace = True)


X_feature = X_feature.swaplevel()
X_feature.sort_index(level = 0, inplace = True)

X_train = X_feature.loc[:'2017-03-01']
X_test = X_feature.loc['2018-03-01':]


# 合併與再整理
idxx = X_feature.xs('2009-03-01', level = 0).index

stock_price = stock_price[idxx]

Y_ExpRet = Y_ExpRet[idxx]
Y_ExpRet = Y_ExpRet.stack().to_frame()
Y_ExpRet.rename(columns = {0:'Y_ExpRet'}, inplace = True)
Y_ExpRet.index = X_train.index

reg_XY = X_train.merge(Y_ExpRet, on = ['Time', 'stock_id'])

# Save
reg_XY.to_csv(path + '整理資料/reg_XY.csv')
stock_price.to_csv(path + '整理資料/Y_test_StockPrice.csv')
X_test.to_csv(path + '整理資料/X_test.csv')

# Read
a = pd.read_csv(path + '整理資料/reg_XY.csv')
b = pd.read_csv(path + '整理資料/Y_test_StockPrice.csv')
c = pd.read_csv(path + '整理資料/X_test.csv')
#%%

### 解釋性資料 ###

import pandas as pd
import numpy as np

path = '/Users/alex_chiang/Documents/迴歸分析/期末_多元迴歸/'

stock = pd.read_csv(path + '原始資料/' + 'stock_price_new.csv', encoding = 'cp950', 
                    parse_dates=True, index_col='年月日')

industry = pd.read_csv(path + '原始資料/' + 'industry.csv', 
                       encoding = 'cp950', index_col='年月')

financial_index = pd.read_csv(path + '原始資料/' + 'untidy_features.csv', 
                              encoding = 'cp950', index_col='年月')

dt1 = [str(industry.index[i])+'31' for i in range(len(industry.index))]
industry.index = pd.to_datetime(dt1)
industry.rename_axis('年月', inplace = True)

dt2 = [str(financial_index.index[i])+'31' for i in range(len(financial_index.index))]
financial_index.index = pd.to_datetime(dt2)
financial_index.rename_axis('年月', inplace = True)
stock.loc['2020-12-31']

# 整理 Y
stock.set_index(['證券代碼'], append = True, inplace = True)
stock = stock.swaplevel(axis = 0)
stock.sort_index(level = 0, inplace = True)
stock.drop(columns = ['簡稱'], inplace = True)
stock.rename(columns = {'收盤價(元)':'price'}, inplace = True)
stock.rename_axis(['stock_id', 'Time'], inplace = True)

stock_price = stock.unstack(0)['price']
stock_price.dropna(axis = 1, inplace = True)
Y_ExpRet = stock_price.pct_change()
Y_ExpRet = Y_ExpRet.iloc[range(1,24,2)]
Y_ExpRet.isnull().values.any()


# 整理 X
X_feature = financial_index.merge(industry, on = ['證券代碼', '簡稱', '年月'])
X_feature.set_index(['證券代碼'], append = True, inplace = True)
X_feature = X_feature.swaplevel(axis = 0)
X_feature.sort_index(level = 0, inplace = True)


X_feature.drop(columns = ['簡稱', '交易所子產業代碼', 
                          'TEJ主產業代碼', 'TEJ子產業代碼'], inplace = True)
X_feature.rename_axis(['stock_id', 'Time'], inplace = True)
X_feature.iloc[:, :-1] = X_feature.iloc[:, :-1].applymap(lambda x: np.nan if x == 'N.A.' else float(x))

idx  = [i for i in X_feature.xs('2009-03-31', level = 1).index if i in Y_ExpRet.columns]
X_feature = X_feature.loc[idx]

for i in idx:
    if len(X_feature.loc[i]) != 13 or X_feature.loc[i].isnull().values.any() != False:
        X_feature.drop(index = i, axis = 0, level = 0, inplace = True)

for i in X_feature.xs('2009-03-31', level = 1).index:
    X_feature.loc[i, '交易所主產業代碼'].replace('N.A.', method = 'ffill', inplace = True)
    X_feature.loc[i, '交易所主產業代碼'].replace('N.A.', method = 'bfill', inplace = True)


X_feature = X_feature.swaplevel()
X_feature.sort_index(level = 0, inplace = True)
X_feature.drop('2009-03-31', inplace = True)


# 合併與再整理
Y_ExpRet = Y_ExpRet[X_feature.xs('2010-03-31', level = 0).index]
Y_ExpRet = Y_ExpRet.stack().to_frame()
Y_ExpRet.rename(columns = {0:'Y_ExpRet'}, inplace = True)
Y_ExpRet.index = X_feature.index
reg_XY = X_feature.merge(Y_ExpRet, on = ['Time', 'stock_id'])
reg_XY.isnull().values.any()

# Save
reg_XY.to_csv(path + '整理資料/reg_XY_new.csv')

# Read
d = pd.read_csv(path + '整理資料/reg_XY_new.csv')
