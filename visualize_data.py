# import thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stock_name = 'BBC'
df = pd.read_csv(stock_name + '_stock.csv').head(30)
plt.subplot(1, 2, 1)
plt.plot(df['<Close>'])
plt.subplot(1, 2, 2)
plt.plot(df['<Volume>'])
plt.show()
