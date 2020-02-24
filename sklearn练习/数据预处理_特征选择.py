"""
特征选择的化包括

1. 过滤法

2. 嵌入法

3. 包装法

"""
import pandas as pd
import numpy as np

#%%
# 加载数据并分出x和y
data = pd.read_csv(r'D:\zone\pycharm_zone\tensor_practice\tensor_practice\sklearn练习\digit recognizor.csv')
print(data.head(5))
print(data.info())
x = data.iloc[:, 1:]
y = data.iloc[:, 0]

#%%
# todo: 1.1 方差过滤
from sklearn.feature_selection import VarianceThreshold
X = VarianceThreshold().fit_transform(x)  # 不填的话默认为零

# 也可以用方差的中位数作为阈值，一次去掉一半的特征
x_fsvar = VarianceThreshold(np.median(x.var())).fit_transform(x)

#若特征是伯努利随机变量，假设p=0.8，即二分类特征中某种分类占到80%以上的时候删除特征
x_bvar = VarianceThreshold(.8 * (1 - .8)).fit_transform(x)

#%%
# KNN 和 随机森林在不同方差过滤效果下的对比
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import cross_val_score

#%%
#======【TIME WARNING：35mins +】======#
knn_res = cross_val_score(KNN(),X,y,cv=5).mean()
print('knn_res, ', knn_res)
#python中的魔法命令，可以直接使用%%timeit来计算运行这个cell中的代码所需的时间
#为了计算所需的时间，需要将这个cell中的代码运行很多次（通常是7次）后求平均值，因此运行%%timeit的时间会
#远远超过cell中的代码单独运行的时间
#======【TIME WARNING：4 hours】======#
# cross_val_score(KNN(),X,y,cv=5).mean()
rf_res = cross_val_score(RFC(n_estimators=10, random_state=0), X, y, cv=5).mean()
print('rf_res,', rf_res)

#%%
# todo: 1.2卡方过滤
#卡方过滤是专门针对离散型标签（即分类问题）的相关性过滤。
# 卡方检验类feature_selection.chi2计算每个非负
#特征和标签之间的卡方统计量，并依照卡方统计量由高到低为特征排名。
# 再结合feature_selection.SelectKBest
#这个可以输入”评分标准“来选出前K个分数最高的特征的类，
# 我们可以借此除去最可能独立于标签，与我们分类目的无关的特征。
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt

#%%
# 根据学习曲线选择k值
score = []
for i in range(390, 200, -10):
    x_fschi = SelectKBest(chi2, k=i).fit_transform(x_fsvar, y)
    once = cross_val_score(RFC(n_estimators=10, random_state=0),
                           x_fschi,
                           y,
                           cv=5,).mean()
    score.append(once)
plt.plot(range(390, 200, -10), score)
plt.show()

x_fschi = SelectKBest(chi2, k=300).fit_transform(x_fsvar, y)
res_fc_fschi = cross_val_score(RFC(n_estimators=10,random_state=0),x_fschi,y,cv=5).mean()
#%%
# 根据p值选择k
chivalue, pvalues_chi = chi2(x_fsvar, y)
k = chivalue.shape[0] - (pvalues_chi > 0.05).sum()
#%%
# todo: 1.3F检验
from sklearn.feature_selection import f_classif
F, pvalues_f = f_classif(x_fsvar, y)
k = F.shape[0] - (pvalues_f > 0.05).sum()
x_fsF = SelectKBest(f_classif, k=k).fit_transform(x_fsvar, y)
res_fsF = cross_val_score(RFC(n_estimators=10,random_state=0),x_fsF,y,cv=5).mean()
#%%
# todo: 1.4互信息法
# 。和F检验相似，它既以做回归也可以做分类，并且包含两个类
# feature_selection.mutual_info_classif（互信息分类）和
# feature_selection.mutual_info_regression（互信息回归）。
# 这两个类的用法和参数都和F检验一模一样，不过互信息法比F检验更加强大，
# F检验只能够找出线性关系，而互信息法可以找出任意关系
from sklearn.feature_selection import mutual_info_classif as MIC
result = MIC(x_fsvar, y)
k = result.shape[0] - sum(result <= 0)
x_fsmic = SelectKBest(MIC, k=k).fit_transform(x_fsvar, y)
res_fsmic = cross_val_score(RFC(n_estimators=10, random_state=0), x_fsmic, y, cv=5).mean()

#%%
# todo: 2.1嵌入法
#SelectFromModel是一个元变换器，
# 可以与任何在拟合后具有coef_，feature_importances_属性或
# 参数中可选惩罚项的评估器一起使用
# （比如随机森林和树模型就具有属性feature_importances_，
# 逻辑回归就带有l1和l2惩罚项，线性支持向量机也支持l2惩罚项）。

# feature_selection.SelectFromModel
# params:
#
#   estimator --- 使用的模型评估器，只要是带feature_importances_或者coef_属性，
#   或带有l1和l2惩罚项的模型都可以使用
#
#   threshold --- 特征重要性的阈值
#
#   prefit --- ？？
#
#   norm_order --- ???
#
#   max_feature --- 阈值设定下的最大的特征数

from sklearn.feature_selection import SelectFromModel

RFC_ = RFC(n_estimators =10,random_state=0)
X_embedded = SelectFromModel(RFC_,threshold=0.005).fit_transform(x,y)
#在这里我只想取出来有限的特征。
# 0.005这个阈值对于有780个特征的数据来说，是非常高的阈值，
# 因为平均每个特征只能够分到大约0.001的feature_importances_
#模型的维度明显被降低了
#同样的，我们也可以画学习曲线来找最佳阈值
#%%
#======【TIME WARNING：10 mins】======#
import numpy as np
import matplotlib.pyplot as plt
# RFC_.fit(X,y).feature_importances_
threshold = np.linspace(0,(RFC_.fit(x,y).feature_importances_).max(),20)
score = []
for i in threshold:
    X_embedded = SelectFromModel(RFC_,threshold=i).fit_transform(x,y)
    once = cross_val_score(RFC_,X_embedded,y,cv=5).mean()
    score.append(once)
plt.plot(threshold,score)
plt.show()
#%%
# 再次进行更精确的学习模型
#======【TIME WARNING：10 mins】======#
score2 = []
for i in np.linspace(0,0.00134,20):
    X_embedded = SelectFromModel(RFC_,threshold=i).fit_transform(x,y)
    once = cross_val_score(RFC_,X_embedded,y,cv=5).mean()
    score2.append(once)
plt.figure(figsize=[20,5])
plt.plot(np.linspace(0,0.00134,20),score2)
plt.xticks(np.linspace(0,0.00134,20))
plt.show()

X_embedded = SelectFromModel(RFC_,threshold=0.000564).fit_transform(x,y)
# X_embedded.shape
cross_val_score(RFC_,X_embedded,y,cv=5).mean()
#=====【TIME WARNING：2 min】=====#
#我们可能已经找到了现有模型下的最佳结果，如果我们调整一下随机森林的参数呢？
cross_val_score(RFC(n_estimators=100,random_state=0),X_embedded,y,cv=5).mean()

# todo: 3.1 wrapper包装法
#class sklearn.feature_selection.RFE (estimator, n_features_to_select=None, step=1, verbose=0)
#参数estimator是需要填写的实例化后的评估器，
#n_features_to_select是想要选择的特征个数，
#step表示每次迭
#代中希望移除的特征个数。
#除此之外，RFE类有两个很重要的属性，
#.support_：返回所有的特征的是否最后被选中的布尔矩阵，
#.ranking_返回特征的按数次迭代中综合重要性的排名。
#类feature_selection.RFECV会在交叉验证循环中执行RFE以找到最佳数量的特征，
#增加参数cv，其他用法都和RFE一模一样。

#======【TIME WARNING: 15 mins】======#
from sklearn.feature_selection import RFE
score = []
for i in range(1,751,50):
    X_wrapper = RFE(RFC_,n_features_to_select=i, step=50).fit_transform(x,y)
    once = cross_val_score(RFC_,X_wrapper,y,cv=5).mean()
    score.append(once)
plt.figure(figsize=[20,5])
plt.plot(range(1,751,50),score)
plt.xticks(range(1,751,50))
plt.show()






