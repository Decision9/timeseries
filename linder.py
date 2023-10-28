import random
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from sklearn.linear_model import LinearRegression

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

def generate_data(beta: np.array, error=0.01,type="default",number=50):
    """ y = a*x0 + b*x1 + c*x2 + e ,  x0 = 1, x1 = t, x2 = t^2

    Args:
        beta (np.array): [a,b,c]
        error (float, optional): 方差. Defaults to 0.01.
        type (str, optional): 参数是否随机. Defaults to "default".
        number (int, optional): 生成数据个数. Defaults to 50.

    Returns:
        T: 生成的解释变量
        Y: 生成的被解释变量
    """
    if type=="random":
        # Generate three random integers as a, b, c between -0.5 and 0.5
        a = np.random.random(1)-0.5
        b = np.random.random(1)-0.5
        c = np.random.random(1)-0.5
        beta = np.vstack((a, b, c))
    beta.reshape(3,1)
    T = np.random.random(number)
    T = np.sort(T)
    T = np.vstack((np.ones(number),T,T**2))
    T = np.transpose(T)
    
    E = [random.gauss(0, error) for _ in range(number)]
    E = np.array(E)
    E = E.reshape(number,1)
    
    Y = np.matmul(T,beta).reshape(number,1) + E
    return T,Y
        


if __name__=='__main__':
    # generate data
    T,Y=generate_data(np.array([0,0,0]),error=0.01,type="random")
  
    beta_fitting, Y_fitting = least_square_model(T,Y)
    print(f"拟合的参数为 {beta_fitting[0]} {beta_fitting[1]} {beta_fitting[2]}")

    fitting_ability = np.dot(np.transpose(Y_fitting),Y_fitting)/np.dot(np.transpose(Y),Y)
    print(f"拟合能力为 {fitting_ability}")

    # 调用python自带工具进行线性回归
    model = LinearRegression().fit(T, Y)
    print("\n")
    print("第三问:")
    print(f"调用线性回归函数后参数为 {model.intercept_[0]} {model.coef_[0][1]} {model.coef_[0][2]}")
    print("\n")
    Y_fitting2 = np.dot(T, np.transpose(model.coef_)) + model.intercept_


    #施密特正交化
    print("施密特正交化")
    T_schmidt = GramSchmidt([Matrix(T[:,0]),Matrix(T[:,1]),Matrix(T[:,2])], True)
    T_schmidt = np.transpose(np.array(T_schmidt)).squeeze()
    T_schmidt = np.array(T_schmidt,dtype=float)
    model_schmidt1, _ = least_square_model(T_schmidt, Y)
    print(f"y = a*x1 + b*x2 调用线性回归函数后参数为 a = {model_schmidt1[0]} {model_schmidt1[1]}, b = {model_schmidt1[2]}")
    model_schmidt2, _ = least_square_model(T_schmidt[:,0:2], Y)
    print(f"y = a*x1 调用线性回归函数后参数为 a = {model_schmidt2[0]} {model_schmidt2[1]}")


    # plt.ylabel('y')
    # plt.xlabel('t')
    # plt.scatter(T[:,1],Y)
    # # 第二问计算结果可视化
    # plt.plot(T[:,1], Y_fitting, color='r')
    # # 第三问计算结果可视化
    # plt.plot(T[:,1], Y_fitting2, color='r')
    # plt.show()
    print('\n')
    A = np.mat([[400,-201],[-800,401]])
    b = np.mat([200,400]).T
    x = np.linalg.solve(A, b)
    lambda_all, _ = np.linalg.eig(A.T @ A)
    print(f'该方程的解为x1={x[0]},x2={x[1]}')
    cond_A = np.sqrt(lambda_all.max()/lambda_all.min())
    print(f'A的条件数为:{cond_A}')
    
    
    #
    # e = np.random.random(1)*0.02-0.01
    e = np.array([np.random.random(1)*0.02-0.01 for _ in range(50)])
    print(e.shape)
    # e = np.transpose(e)
    x1 = np.random.random(50)
    x1 = np.vstack((x1))
    # x1 = np.transpose(x1)
    print(x1.shape)
    x2 = x1+1+e
    print(x2.shape)
    model_p, y_p = least_square_model(x1, x2)
    print('\n')
    print(model_p)
    error = 1/(1- np.dot(np.transpose(y_p),y_p)/np.dot(np.transpose(x2),x2))
    print(f"方差膨胀因子 {error}")
    
    
    print("--------------------------第五次作业-----------------------------------")
    print('第一问：')
    T,Y=generate_data(None,error=0.01,type="random")
    beta_fit, Y_hate = least_square_model(T,Y)
    
    #参差平方和
    e_square = (Y-Y_hate).T@(Y-Y_hate)
    print(f"参差平方和为：{e_square}")
    
    #sigma^2的无偏估计
    sigma_square = e_square/(50-3)
    print(f"sigma^2的无偏估计为：{sigma_square}")
    
    #拟合优度
    r_uc = 1-e_square/(Y.T@Y)
    r_c = 1-e_square/(np.sum((Y-np.mean(Y))**2))
    r_j = 1-(e_square/(50-3))/(np.sum((Y-np.mean(Y))**2)/(50-1))
    print(f"三种拟合优度分别为:R_uc={r_uc},R_c={r_c},R_j={r_j}")
    
    #假设检验 t 检验
    p_value = (beta_fit[1]-0)/np.sqrt((np.linalg.inv(T.T@T))[1,1]*(sigma_square))
    print(f"第一种检验的p值估计为：{p_value},t检验(自由度47)对应的范围是[-2.945，2.945]")
    
    w_value = 2.945*np.sqrt((np.linalg.inv(T.T@T))[1,1]*(sigma_square))
    print(f"第二种检验显著水平关键值为：{w_value}，而beta_1为:{beta_fit[1]}")
    
    print('\n第二问：')
    model = LinearRegression().fit(T, Y)
    print(f"统计软件自带的模型拟合系数为：{model.score(T,Y)}, 为第二种拟合系数。")
    
    print('------------------------第六次作业--------------------------')
    T,Y = generate_data(np.array([0.8,1,0.5]))
    beta_fitting, Y_fitting = least_square_model(T,Y)
    print(f"回归的参数为 {beta_fitting[0]} {beta_fitting[1]} {beta_fitting[2]}")
  
    #构造F统计量 r=1 n=50 k=3
    r=1 
    n=50 
    k=3

    print('\n假设检验alpha1+alpha2=1.5')
    #alpha1+alpha2=1.5
    #Rb-q
    S_square = ((Y-Y_fitting).T@(Y-Y_fitting))/(n-k)
    R = np.array([0,1,1]).reshape(1,3)
    q = 1.5
    w = (R@beta_fitting-q).T@np.linalg.inv(R@np.linalg.inv(T.T@T)@R.T)@(R@beta_fitting-q)
    # print(np.linalg.inv(R@np.linalg.inv(T.T@T)@R.T))
    F = (w/r)/S_square
    print(f'直接构造F统计量为：{F}，小于F(1,47)在alpha=5%、10%的检测值，故接受该假设')
    
    #利用约束构造统计量
    #此时回归方程变为Y=alpha0+(1.5-alpha2)*T+alpha2*T^2, 等价于Y-1.5*T=alpha0+alpha2*(T^2-T)
    Y_heta = Y.reshape(n,1) - 1.5*T[:,1].reshape(n,1)
    T_heta = np.hstack((T[:,0].reshape(n,1),T[:,2].reshape(n,1)-T[:,1].reshape(n,1)))
    beta_heta_fitting, Y_heta_fitting = least_square_model(T_heta,Y_heta)
    print(f"修改后的模型回归的参数为 {beta_heta_fitting[0]} {beta_heta_fitting[1]}")    
    F_heta = ((Y_heta-Y_heta_fitting).T@(Y_heta-Y_heta_fitting)-(Y-Y_fitting).T@(Y-Y_fitting))/r / S_square
    print(f'直接构造F统计量为：{F_heta}，小于F(1,47)在alpha=5%、10%的检测值，故接受该假设')
    
    
    print('\n假设检验alpha1+alpha2=10')
    #alpha1+alpha2=10
    #Rb-q
    S_square = ((Y-Y_fitting).T@(Y-Y_fitting))/(n-k)
    R = np.array([0,1,1]).reshape(1,3)
    q = 10
    w = (R@beta_fitting-q).T@np.linalg.inv(R@np.linalg.inv(T.T@T)@R.T)@(R@beta_fitting-q)
    # print(np.linalg.inv(R@np.linalg.inv(T.T@T)@R.T))
    F = (w/r)/S_square
    print(f'直接构造F统计量为：{F}，大于F(1,47)在alpha=5%、10%的检测值，故拒绝该假设')
    
    #利用约束构造统计量
    #此时回归方程变为Y=alpha0+(10-alpha2)*T+alpha2*T^2, 等价于Y-10*T=alpha0+alpha2*(T^2-T)
    Y_heta = Y.reshape(n,1) - 10*T[:,1].reshape(n,1)
    T_heta = np.hstack((T[:,0].reshape(n,1),T[:,2].reshape(n,1)-T[:,1].reshape(n,1)))
    beta_heta_fitting, Y_heta_fitting = least_square_model(T_heta,Y_heta)
    print(f"修改后的模型回归的参数为 {beta_heta_fitting[0]} {beta_heta_fitting[1]}")    
    F_heta = ((Y_heta-Y_heta_fitting).T@(Y_heta-Y_heta_fitting)-(Y-Y_fitting).T@(Y-Y_fitting))/r / S_square
    print(f'直接构造F统计量为：{F_heta}，大于F(1,47)在alpha=5%、10%的检测值，故拒绝该假设')