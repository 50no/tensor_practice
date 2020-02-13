import numpy as np
# t = np.arange(24).reshape((4, 6))

# todo:小于10数据修改为0
# t[t<10] = 0
# print(t)

# todo:三目运算符
# t = np.where(t<10, 0, 10) # 小10的置零，否则置10
# print(t)

# 如果想要小于10的置10，大于18的置18怎么办，
# todo:这就需要用到“剪裁”！
# t = t.clip(10, 18)
# # print(t)

# todo:将nan转换成0
# t[np.isnan(t)] = 0

# todo:垂直合并和水平合并
# t1 = np.arange(20).reshape((2, 10))
# t2 = np.arange(19, -1, -1).reshape((2, 10))
#
# test = np.vstack((t1, t2))
# test02 = np.hstack((t1, t2))

# todo: 交换行和列
# t = np.arange(12, 24).reshape(3, 4)
# print(t)
# t[[1, 2], :] = t[[2,1], :]
# print(t)


# todo:各种函数
# 求和：t.sum(axis=None)
# 均值：t.mean(a,axis=None)  受离群点的影响较大
# 中值：np.median(t,axis=None)
# 最大值：t.max(axis=None)
# 最小值：t.min(axis=None)
# 极值：np.ptp(t,axis=None) 即最大值和最小值只差
# 标准差：t.std(axis=None)





