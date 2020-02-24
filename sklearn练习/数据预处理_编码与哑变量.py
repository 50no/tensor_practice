"""
分为三部分：
1. 标签的转换为数值型的方法

2. 特征转换为数值型的方法

3. 独热编码

"""
import pandas as pd
# import printallvalues

data = pd.read_csv(r'tensor_practice/sklearn练习/Narrativedata.csv',index_col=0)
# 这里不用.values.mean()是因为numpy的mean（）会将nan也算进去，而pandas不会
data.loc[:, 'Age'] = data.loc[:, 'Age'].fillna(data.loc[:, 'Age'].mean())
data.dropna(how='any', axis=0, inplace=True)
data.loc[:, 'Survived'] = data.loc[:, 'Survived'].where(data.loc[:, 'Survived']!='Unknown')
data.loc[:, 'Survived'].fillna('No', inplace=True)
data1 = data.copy()
data2 = data.copy()
print(data.head(10))

# todo: 1.标签的转换为数值型的方法
from sklearn.preprocessing import LabelEncoder
data.loc[:, 'Survived'] = LabelEncoder().fit_transform(data.loc[:, 'Survived'])
print(data.head(10))
print(LabelEncoder().fit(data.loc[:, 'Survived']).classes_)


# todo: 2.特征转换为数值型的方法
from sklearn.preprocessing import OrdinalEncoder
data1.iloc[:, 1:3] = OrdinalEncoder().fit_transform(data1.iloc[:, 1:3])
print(data1)


# todo: 3.独热编码
from sklearn.preprocessing import OneHotEncoder
one_hot_array = OneHotEncoder(categories='auto').fit_transform(data2.iloc[:, 1:-1]).toarray()
print(one_hot_array)
data2.drop(columns=['Sex', 'Embarked'], axis=1, inplace=True)
data2 = pd.concat([data2, pd.DataFrame(one_hot_array)], axis=1)
data2.columns = ['Age', 'Survived', 'female', 'male', 'embarked_c', 'embarked_q', 'embarked_s']
print(data2.head(10))

"""查看属性为 get_feature_names"""






