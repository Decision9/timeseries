import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
plt.rcParams['font.sans-serif']= ['SimHei']
x = np.linspace(0,8*np.pi,100).reshape(100,1)
e = np.array([np.random.random(1)*0.5 for _ in range(100)]).reshape(100,1)
#k=1
y = x+np.sin(x)+e

#趋势项
model = LinearRegression().fit(x, y)
y_hate = np.dot(x, np.transpose(model.coef_)) + model.intercept_

#循环项
x_circles = []
for i in range(25):
    mean_x = (y[i]-y_hate[i] + y[i+25]-y_hate[i+25] + y[i+50]-y_hate[i+50] + y[i+75]-y_hate[i+75])/(2*np.pi)
    x_circles.append(mean_x)
    
x_circles = np.array(x_circles).reshape(25,1)
x_circles = np.tile(x_circles,(4,1))

plt.ylabel('y')
plt.xlabel('x')
# plt.scatter(x,y)
# 第二问计算结果可视化
plt.plot(x, y, color='c',label='初始数据')
plt.plot(x, y_hate, color='m',ls='--',label='趋势项')
plt.plot(x,x_circles, color='r',label='循环项')
plt.scatter(x,y-y_hate-x_circles,color='g',s=3,label='误差项')
plt.legend()
plt.show()

