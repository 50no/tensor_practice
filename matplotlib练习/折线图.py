from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] =False  # 显示负号
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签

x = list(range(11, 31))
y = [1,2,1,1,2,4,3,2,3,4,4,5,6,5,4,3,3,1,1,1,]
y2 =[1,2,2,3,4,5,5,6,8,7,6,5,5,4,3,2,2,1,0,1,]

plt.figure(figsize=(15, 7), dpi=80)
plt.plot(x, y, label='我', color='r', linestyle='--', linewidth=5, alpha=0.8)
plt.plot(x, y2, label='儿子')
xticks_labels = ['{}岁'.format(i) for i in x]
plt.yticks(size=19)
plt.xticks(x, xticks_labels, rotation=45, size=19)
plt.grid(alpha=0.4)
plt.legend(loc='upper left', fontsize=18)

plt.show()









