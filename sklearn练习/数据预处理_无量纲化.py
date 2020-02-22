import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]

d_f_m = pd.DataFrame(data)
d_f_s = pd.DataFrame(data)

d_f_m = MinMaxScaler(feature_range=[0, 1]).fit_transform(d_f_m)
d_f_s = StandardScaler().fit_transform(d_f_s)

"""
1. 单步fit得到的结果可通过.inverse_transform(new_data)得到原始数据
2. StandarScaler单步fit的结果可以用 .mean_ .var_查看属性
"""




