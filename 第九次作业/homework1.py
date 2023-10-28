import numpy as np
    
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
    x = generate_data(np.array([1,-0.5]),np.array([0.3,0.4]),100)
    print(x)