import pandas as pd
import numpy as np

a=np.array([[1,2,3],[4,3,6],[7,8,9]])
df1=pd.DataFrame(a,index=['row0','row1','row2'],columns=list('ABC'))
df2 =df1.copy()
df3 =df1.copy()
df4 =df1.copy()
print(df1)
print('*'*99)

# # todo: 1.删除C列中有3的所有行
# d_index = df1.loc[:, 'C'][df1.loc[:, 'C']==3].index
# df1.drop(index=d_index, inplace=True)
# print(df1)
#
# # todo: 2. 删除row0行中有3的所有列
# d_columns = df2.loc['row0', :][df2.loc['row0', :]==3].index
# # print(d_columns)
# df2.drop(columns=d_columns, inplace=True)
# print(df2)

# todo: 1.删除C列中有3的所有行
df1.loc[:, 'C'] = df1.loc[:, 'C'].where(df1.loc[:, 'C'] != 3)
df1.dropna(axis=0, inplace=True)
print(df1)

# todo: 2. 删除row0行中有3的所有列
df2.loc['row0', :] = df2.loc['row0', :].where(df2.loc['row0', :]!=3)
df2.dropna(axis=1, inplace=True)
print(df2)

# todo: 3.删除所有有3的行
df3 = df3.where(df3!=3)
df3.dropna(axis=0, inplace=True)
df3 = df3.astype(np.int32)
print(df3)

# todo: 4.删除所有有3的行
df4 = df4.where(df4!=3)
df4.dropna(axis=1, inplace=True)
print(df4)





