import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.api import qqplot
warnings.filterwarnings('ignore')

def generate_data(p:np.array, q:np.array, T):
    """
    生成ARIMA(p, d, q)模型的数据, 暂不考虑差分d，在该函数生成的数据默认d=0。
    Args:
        p (np.array): p=[theta1, theta2, ......], p的长度为AR模型阶数，在该函数表现为阶数的参数
        q (np.array): q=[beta1, beta1, ......], q的长度为MA模型阶数, 在该函数表现为阶数的参数
        T :生成的数据量
    """
    if T > max(len(p),len(q)):
        equ_1 = [-1] + p.tolist()
        equ_1.reverse()
        equ_2 = [1] + q.tolist()
        equ_2.reverse()
        root_1 = np.roots(equ_1)
        root_2 = np.roots(equ_2)
        if (abs(root_1).min()>1) & (abs(root_2).min()>1):
            True
        else:
            print(f'方程的根为{root_1}、{root_2}, 其最小长度为{min(abs(root_1).min(),abs(root_2).min())}')
            print('给定的参数无法使模型平稳')
            pass
    else:
        print('T的值必须大于max(p,q)')
        pass
    
    a_t = np.random.randn(T+len(q))
    x_t = np.random.random(T+len(p))

    for i in range(len(p),T+len(p)):
        pre_x = x_t[i-len(p):i]
        pre_a = a_t[i-len(q):i]
        x_t[i] = np.dot(p,pre_x[::-1])+a_t[i]+np.dot(q,pre_a[::-1])
    return x_t
    
if __name__=='__main__':
    # generate different level data
    x_1 = generate_data(np.array([0.5]),np.array([0.5]),100)
    x_3 = generate_data(np.array([1,-0.5,0.2]),np.array([0.3,0.4,0.1]),100)
    x_5 = generate_data(np.array([1,-0.5,0.2,-0.3,0.2]),np.array([0.1,0.7,0.3,-0.5,-0.1]),100)  
    
    # 定阶
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(321)
    fig = plot_acf(x_1, lags=10, ax=ax1, title='x_1 ACF')
    ax2 = fig.add_subplot(322)
    fig = plot_pacf(x_1,lags=10, ax=ax2, title='x_1 PACF')
    ax3 = fig.add_subplot(323)
    fig = plot_acf(x_3, lags=10, ax=ax3, title='x_3 ACF')
    ax4 = fig.add_subplot(324)
    fig = plot_pacf(x_3,lags=10, ax=ax4, title='x_3 PACF')
    ax5 = fig.add_subplot(325)
    fig = plot_acf(x_5, lags=10, ax=ax5, title='x_5 ACF')
    ax6 = fig.add_subplot(326)
    fig = plot_pacf(x_5,lags=10, ax=ax6, title='x_5 PACF')
    # plt.show()
    
    #模型拟合，估计参数，估计残差
    resid = []
    predict = []
    data = [x_1,x_3,x_5]
    for i in [1,3,5]:
        x = data.pop(-1)
        armodel = ARIMA(x,order=(i,0,i)).fit()
        predict.append(armodel.predict(0))
        resid.append(armodel.resid)

    # 残差图、QQ图检验、模型预测
    fig1 = plt.figure(figsize=(12,10))
    plt.subplot(321)
    plt.plot(resid[0], label = '混合模型(1,0,1)')
    plt.title('混合模型(1,0,1) resid')
    plt.subplot(322)
    plt.plot(x_1,color='g',label='源数据')
    plt.plot(predict[0],color='b',label='预测项')
    plt.legend()
    
    plt.subplot(323)
    plt.plot(resid[1], label = '混合模型(3,0,3)')
    plt.title('混合模型(3,0,3) resid')
    plt.subplot(324)
    plt.plot(x_3,color='g',label='源数据')
    plt.plot(predict[1],color='b',label='预测项')
    plt.legend()
    
    plt.subplot(325)
    plt.plot(resid[2], label = '混合模型(5,0,5)')
    plt.title('混合模型(5,0,5) resid')
    plt.subplot(326)
    plt.plot(x_5,color='g',label='源数据')
    plt.plot(predict[2],color='b',label='预测项')
    plt.legend()

    fig2 = plt.figure(figsize=(12,8))
    fig2.tight_layout(h_pad=8)
    ax = fig2.add_subplot(311)
    fig2 = qqplot(resid[0], line='q', ax=ax, fit=True)
    plt.title('1-level data QQ plot')
    ax1 = fig2.add_subplot(312)
    fig2 = qqplot(resid[1], line='q', ax=ax1, fit=True)
    plt.title('3-level data QQ plot')
    ax2 = fig2.add_subplot(313)
    fig2 = qqplot(resid[2], line='q', ax=ax2, fit=True)
    plt.title('5-level data QQ plot')
    plt.show()

    
