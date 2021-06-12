import pandas as pd
import os
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

os.chdir("/Users/alex_chiang/Documents/GitHub/RegressionAnalysis-Annual-Return-and-Financial/")
path = os.getcwd()

# Only Continuous Variable
df = pd.read_csv(path + '/Tidy_Data/reg_XY_new.csv')
df.drop(columns = ['stock_id', 'Time'], inplace = True)

x = sm.add_constant(df.drop(columns = ['交易所主產業代碼', 'Y_ExpRet']))
Y = df['Y_ExpRet']

reg_model = sm.OLS(Y, x)
result = reg_model.fit()
result.summary()


# Calculate VIF
# https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python

VIF = pd.DataFrame([variance_inflation_factor(x.values, i) 
                    for i in range(x.shape[1])], 
                    index=x.columns, columns = ['VIF'])


# Add Dummy Variable
# https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html

dummy = pd.get_dummies(df['交易所主產業代碼'])

df_dummy = df.drop(columns = ['交易所主產業代碼', 'Y_ExpRet'])
df_dummy = pd.concat([df_dummy, dummy], axis = 1)

x = sm.add_constant(df_dummy)
Y = df['Y_ExpRet']

reg_model_dummy = sm.OLS(Y, x)
result_dummy = reg_model_dummy.fit()
result_dummy.summary()



