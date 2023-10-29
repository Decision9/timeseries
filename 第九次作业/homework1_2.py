import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.arima.model.ARIMA import
from statsmodels.tsa.stattools import adfuller as ADF
import statsmodels as sm
from scipy import stats

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import warnings
warnings.filterwarnings('ignore')

work_path = os.path.dirname(os.path.abspath(__file__))
# dataparse = lambda dates: pd.datetime.strptime(dates,'%Y-%m')
# data = pd.read_csv(work_path+'/Data.xlsx',parse_dates=['Month'],index_col='Month',data_parser=dataparse)
data = pd.read_excel(work_path+'/Data.xlsx')
# data=data[~data.isin (nan).any(axis=1)]
data.drop('中国:GDP:现价:当季值',axis=1,inplace=True)
print(ADF(data['中国:固定资产投资完成额:累计值']))


data['diff_1'] = data['中国:固定资产投资完成额:累计值'].diff(1)
data = data[data['diff_1'].notna()]
print(ADF(data['diff_1']))
# print(data['diff_1'])
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = plot_acf(data['中国:固定资产投资完成额:累计值'], lags=30, ax=ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(data['中国:固定资产投资完成额:累计值'],lags=30, ax=ax2)
# plt.show()

print('以PACF来看，模型大概为2阶。')

pmax = 5
qmax = 5
bic_matrix =[]#bic矩阵
for p in range(pmax+1):
    tmp = []
    for q in range(qmax+1):#存在部分报错，所以用try来跳过报错。
        try:
            tmp.append(ARIMA(data['中国:固定资产投资完成额:累计值'],order=(p,1,q)).fit().bic)
        except:
            tmp.append(None)
    bic_matrix.append(tmp)
bic_matrix = pd.DataFrame(bic_matrix)#从中可以找出最小值
p,q = bic_matrix.stack( ).idxmin()
##先用stack展平，然后用idxmin找出最小值位置。
print('--------------------------------------------------------')
print(u'BIc最小的p值和q值为: %s、%s' %(p,q))
print('利用混合模型定阶后为(3,0,3)')

armodel303 = ARIMA(tuple(data['中国:固定资产投资完成额:累计值']),order=(3,0,3)).fit()
resid = armodel303.resid
# print(dir(resid))
plt.figure(figsize=(12,8))
plt.plot(resid, label = '混合模型(3,0,3)')
plt.show()

print('该数据模型拟合效果较差.')

