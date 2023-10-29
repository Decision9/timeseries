import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
from arch import arch_model
work_path = os.path.dirname(os.path.abspath(__file__))

#上证指数
# data = pd.read_excel(work_path+'/000001.SH.xlsx')
#贵州茅台
data = pd.read_excel(work_path+'/600519.SH.xlsx')

data['收益率'] = data['收盘价(元)']/data['收盘价(元)'].shift(1)-1
data = data[data['收益率'].notna()]
data_std = np.std(data['收益率'])
print(data_std)
print(data['收益率'])

am = arch_model(data['收益率'])
model2=am.fit(update_freq=0) #估计参数

#模型的拟合效果为
print(model2.summary())