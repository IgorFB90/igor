import pandas_datareader.data as web
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
start = dt.datetime(2017, 4, 1)
end = dt.datetime.today()

data = web.DataReader("AAPL", 'google', start, end)
print(data.head()) 

data[['High','Low']].plot()
plt.legend()
plt.show()