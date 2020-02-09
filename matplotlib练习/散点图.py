import random
from matplotlib import pyplot as plt

import matplotlib
matplotlib.rcParams['axes.unicode_minus'] =False  # 显示负号
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签

y_3 = [ random.randint(1, 30) for _ in range(31)]
y_10 = [ random.randint(1, 30) for _ in range(31)]

x_3 = list(range(1, 32))
x_10 = list(range(51, 82))

plt.figure(figsize=(20, 8), dpi=80)

plt.scatter(x_3[::2], y_3[::2], label='三月')
plt.scatter(x_10[::2], y_10[::2], label='十月')

xticks_labels = ['3月{}日'.format(i) for i in range(1, 32)]
xticks_labels += ' '
xticks_labels += ['10月{}日'.format(i) for i in range(1, 32)]
plt.xticks((x_3[::2] + x_10[::2]), xticks_labels[::2], size=15, rotation=270)

plt.xlabel('时间', size=19)
plt.ylabel('温度', size=19)
plt.title('标题', size=29)

plt.legend(loc='upper left', fontsize=12)
plt.grid(alpha=0.5, axis='y')

plt.show()


