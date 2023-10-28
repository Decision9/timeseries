import os
import numpy as np
import sys
import pandas as pd
work_path = os.path.dirname(os.path.abspath(__file__))
data_path = work_path+'/MCdata_SH(1).csv'

def least_square_model(X, Y):
    """最小二乘法估计调用
    Args:
        X (np.array): 解释变量，n*k大小
        Y (np.array): 被解释变量
    output:
        beta(np.array): 被估计参数值
        Y_fitting: 被估计的y值
    """
    
    beta = np.dot(np.linalg.inv(np.dot(np.transpose(X),X)), np.dot(np.transpose(X),Y))
    Y_fitting = np.dot(X,beta)
    return beta,Y_fitting

data = pd.read_csv(data_path,header='infer',sep=',')
#计算实际GDP
data['GDP'] = data['GDP']/data['CPI']
#计算实际投资
data['Invest'] = data['Invest']/data['CPI']
#计算通货膨胀率，并将其改为百分制
data['P'] = ((data['CPI']-data['CPI'].shift(1))/data['CPI'].shift(1))*100
data['year'] = data['year']-1989
Y = np.array(data['Invest']).reshape(29,1)
X = np.append(np.array(data[['year','GDP','R','P']]),np.ones(29).reshape(29,1),axis=1)
print(X)
Y = Y[1:,:]
X = X[1:,:]
beta_fitting, Y_fitting = least_square_model(X,Y)
print(f"拟合的参数为 {beta_fitting[0]}*year + {beta_fitting[1]}*GDP + {beta_fitting[2]}*R + {beta_fitting[3]}*P + {beta_fitting[4]}")

fitting_ability = np.dot(np.transpose(Y_fitting),Y_fitting)/np.dot(np.transpose(Y),Y)
print(f"拟合能力为 {fitting_ability}")
