import pandas as pd
import numpy as np

a=np.array([[1,2,3],[4,3,6],[7,8,9]])
df1=pd.DataFrame(a,index=['row0','row1','row2'],columns=list('ABC'))
df2=df1.copy()
df3 = df1.copy()
df4 = df1.copy()
print(df1)
print('*'*99)

# todo：1.C列中所有等于3的元素修改为999
df1.loc[:, 'C'][df1.loc[:, 'C']==3] = 999
print(df1)

# todo：2.所有列等于3的元素修改为999
for column in df2.columns:
   df2.loc[:, column][ df2.loc[:, column]==3] = 999
print(df2)

# todo：3.row0行所有等于3的元素修改为999
print(df3.loc['row0', :]==3)
# df3.loc['row0', [df3.loc['row0', :]==3].] = 999
# print(df3)


# df4.loc[:, 'C'] = df4.loc[:, 'C'][df4.loc[:, 'C']!=3]
# print(df4)