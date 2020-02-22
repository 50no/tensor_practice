import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation   #导入负责绘制动画的接口
#其中需要输入一个更新数据的函数来为fig提供新的绘图信息

fig, ax = plt.subplots()          #生成轴和fig,  可迭代的对象
#生成子图，相当于fig = plt.figure(),ax = fig.add_subplot(),其中ax的函数参数表示把当前画布进行分割，
# 例：fig.add_subplot(2,2,2).表示将画布分割为两行两列
x, y= [], []    #用于接受后更新的数据
line, = plt.plot([], [], '.-')   #绘制线对象，plot返回值类型，要加逗号

#------说明--------#
#核心函数包含两个：
#一个是用于初始化画布的函数init()
#另一个是用于更新数据做动态显示的update()


def init():
	#初始化函数用于绘制一块干净的画布，为后续绘图做准备
    ax.set_xlim(-5, 15*np.pi)    #初始函数，设置绘图范围
    ax.set_ylim(-3, 3)
    return line

def update(step):           #通过帧数来不断更新新的数值
    x.append(step)
    y.append(np.cos(step/3)+np.sin(step**2))    #计算y
    line.set_data(x, y)
    return line

#fig 是绘图的画布
#update 为更新绘图的函数，step数值是从frames 传入
#frames 数值是用于动画每一帧的数据
ani = FuncAnimation(fig, update, frames=np.linspace(0, 13*np.pi, 128),
                    init_func=init,interval=20)

plt.show()
