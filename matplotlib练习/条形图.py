from matplotlib import pyplot as plt

import matplotlib
matplotlib.rcParams['axes.unicode_minus'] =False  # 显示负号
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签

a = ['星球大战3', '霍顿时刻', '蜘蛛侠：英雄远征', '战狼2']
b_16 = [15746, 312, 4497, 319]
b_15 = [12357, 157, 2045, 168]
b_14 = [2358, 399, 2358, 362]

bar_width = 0.2
x_14 = list(range(len(a)))
x_15 = [i+bar_width for i in x_14]
x_16 = [i+bar_width*2 for i in x_14]

plt.figure(figsize=(12, 8), dpi=80)

plt.bar(x_14, b_14, width=bar_width, label='9月1日', color='black')
plt.bar(x_15, b_15, width=bar_width, label='9月2日')
plt.bar(x_16, b_16, width=bar_width, label='9月3日')
plt.legend()

plt.xticks(x_15, a)

plt.show()

