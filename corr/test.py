import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

# 9月26日无上午图片，10月26日无下午图片
dataframe = pd.read_excel(r'excel\temp_all.xlsx')
dataframe = dataframe[['IMG_ID', 'BIQME', 'PM2.5']]
print(dataframe.head())

# 近看上午角度1的，最后日期11月16
# am1 = dataframe.head(51)
am1 = dataframe[dataframe['IMG_ID'].str.contains('*1.jpg')]
print(am1)
# am1.plot(x='IMG_ID')
# plt.show()

matrix = am1.corr(method='spearman')
print(matrix)
arr = np.array(am1[['BIQME', 'PM2.5']])
corr, p_val = stats.spearmanr(arr)
print(corr)
print(p_val)