from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

data = load_breast_cancer()
# print(data.data.shape)
# print(data.target.shape)
rfc = RandomForestClassifier(n_estimators=100, random_state=90)
score_pre = cross_val_score(rfc, data.data, data.target, cv=10)
# print(score_pre)

# 网格搜索应该在学习曲线之后进行，因为学习曲线可以根据图看到规律
score_1ist = []
for i in range(0, 200, 10):
    rfc = RandomForestClassifier(n_estimators=i+1,
                                 n_jobs=-1,
                                 random_state=90,)
    score = cross_val_score(rfc, data.data, data.target, cv=10).mean()
    score_1ist.append(score)
print(max(score_1ist), (score_1ist.index(max(score_1ist))*10+1))
fig = plt.figure(figsize=[12, 3])
plt.plot(range(1, 201, 10), score_1ist)
plt.show()

scorel = []
for i in range(65,75):
    rfc = RandomForestClassifier(n_estimators=i,
                                 n_jobs=-1,
                                 random_state=90)
    score = cross_val_score(rfc,data.data,data.target,cv=10).mean()
    scorel.append(score)
print(max(scorel),([*range(35,45)][scorel.index(max(scorel))]))
fig = plt.figure(figsize=[12,3])
plt.plot(range(65,75),scorel)
plt.show()

"""
有一些参数是没有参照的，很难说清一个范围，这种情况下我们使用学习曲线，看趋势
从曲线跑出的结果中选取一个更小的区间，再跑曲线
param_grid = {'n_estimators':np.arange(0, 200, 10)}

param_grid = {'max_depth':np.arange(1, 20, 1)}

param_grid = {'max_leaf_nodes':np.arange(25,50,1)}
    对于大型数据集，可以尝试从1000来构建，先输入1000，每100个叶子一个区间，再逐渐缩小范围

有一些参数是可以找到一个范围的，或者说我们知道他们的取值和随着他们的取值，模型的整体准确率会如何变化，这
样的参数我们就可以直接跑网格搜索
param_grid = {'criterion':['gini', 'entropy']}

param_grid = {'min_samples_split':np.arange(2, 2+20, 1)}

param_grid = {'min_samples_leaf':np.arange(1, 1+10, 1)}

param_grid = {'max_features':np.arange(5,30,1)} 

"""

# 调整max_depth的深度
param_grid = {'max_depth': np.arange(1, 20, 1)}
rfc = RandomForestClassifier(n_estimators=73,
                             random_state=90,)

GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(data.data, data.target)

print('Best_params: {}'.format(GS.best_params_))
print('Best_score: {}'.format(GS.best_score_))



