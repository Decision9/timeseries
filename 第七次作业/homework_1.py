import numpy as np
import pandas as pd
import os
work_path = os.path.dirname(os.path.abspath(__file__))
#上证指数
# data = pd.read_excel(work_path+'/000001.SH.xlsx')
#沪深300
# data = pd.read_excel(work_path+'/000300.SH.xlsx')
#贵州茅台
data = pd.read_excel(work_path+'/600519.SH.xlsx')
#豫园股份
# data = pd.read_excel(work_path+'/600655.SH.xlsx')

data['对数收益率'] = np.log(data['收盘价(元)']/data['收盘价(元)'].shift(1))
print(f'收盘价的对数收益率为：{data["对数收益率"]}')
print(f'中位数为：{np.median(data["收盘价(元)"])}, 均值为：{np.mean(data["收盘价(元)"])}, 偏度为：{data["收盘价(元)"].skew()}， 峰度为：{data["收盘价(元)"].kurtosis()}')
