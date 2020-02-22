"""
1. 二值化

2. 分段

"""


import pandas as pd
import printallvalues
import numpy as np

data = pd.read_csv(r'./Narrativedata.csv',index_col=0)
data.loc[:, 'Age'] = data.loc[:, 'Age'].fillna(data.loc[:, 'Age'].mean())
data.dropna(axis=0, how='any', inplace=True)
data1 = data.copy()
print(data.info())

# todo: 1.二值化
from sklearn.preprocessing import Binarizer
x = data.iloc[:, 0].values.reshape(-1, 1)
transformer = Binarizer(threshold=30).fit_transform(x)
print(type(transformer))

# todo: 2.分段
"""
preprocessing.KBinsDiscretizer
params:
    n_bins: 桶的数量
    
    encode: 编码方式
        onehot, 独热编码
        ordinal, 就是普通的数值编码
    
    strategy:
        uniform, 等宽分箱（根据值）
        quantile, 等位分箱（根据位置）
        kmeans, 按聚类分箱
"""
from sklearn.preprocessing import KBinsDiscretizer
x = data.iloc[:, 0].values.reshape(-1, 1)
res_ord = KBinsDiscretizer(n_bins=3,
                           encode='ordinal',
                           strategy='uniform',).fit_transform(x).astype(np.int32)
print(type(res_ord))
print(set(res_ord.ravel()))  # 这里使用ravel是为了降维，否则set不了

res_one = KBinsDiscretizer(n_bins=3,
                           encode='onehot',
                           strategy='uniform',).fit_transform(x).toarray().astype(np.int32)
print(res_one)










