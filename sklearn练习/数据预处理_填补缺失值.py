import pandas as pd
import printallvalues


data_sk = pd.read_csv(r'./Narrativedata.csv',index_col=0)
data_pd = pd.read_csv(r'./Narrativedata.csv',index_col=0)
print(data_pd.info())
# print(data_pd)


# todo:第一种方法，利用sklearn的内置接口
from sklearn.impute import SimpleImputer


# 这里填补AGE
Age = data_sk.loc[:, 'Age'].values.reshape(-1, 1)

imp_mean = SimpleImputer(strategy='mean')
imp_median = SimpleImputer(strategy='median')
imp_0 = SimpleImputer(strategy='constant', fill_value=0)

data_sk.loc[:, 'Age'] = imp_mean.fit_transform(Age)
# imp_median = imp_median.fit_transform(Age)
# imp_0 = imp_0.fit_transform(Age)
print(data_sk.info())

# todo: 第二种方法，利用pandas
data_pd.loc[:, 'Age'] = data_pd.loc[:, 'Age'].fillna(data_pd.loc[:, 'Age'].median())
print(data_pd.info())