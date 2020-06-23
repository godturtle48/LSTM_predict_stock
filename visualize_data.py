# import thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stock_name = 'CMG'
date_range = 30
df = pd.read_csv(stock_name + '_stock.csv', index_col='<Date>').head(date_range)
# df = pd.read_csv('./CafeF.HNX.Upto26.05.2020.csv', index_col='<Date>')
# companyDf = df[df['<Ticker>'] == stock_name].head(date_range)
# companyDf = companyDf.iloc[::-1]
df = df.iloc[::-1]
plt.subplot(1, 2, 1)
plt.title(stock_name + ' close')
plt.plot(df['<Close>'])
plt.xticks(rotation=90)
plt.subplot(1, 2, 2)
plt.title(stock_name + ' volume')
plt.plot(df['<Volume>'])
plt.xticks(rotation=90)
plt.show()
