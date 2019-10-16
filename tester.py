from feature_functions import *
import pandas as pd
import plotly as py
from plotly import tools
import plotly.graph_objs as go

# (1)Load Data and Create Moving Average

df = pd.read_csv("data\EurUsdHour.csv")
df.columns = ["date", "open", "high", "low", "close", "AskVol"]
df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y %H:%M:%S.%f")
df.set_index("date", inplace=True)
df = df.drop_duplicates(keep=False)
df = df.iloc[:200]

ma = df.close.rolling(center=False, window=30).mean()

# (2)Get Function Data From Selected Functions

HAresults = heikenashi(df,[1])
HA = HAresults.candles[1]

# resampled = OHLCresample(df,'15H')
# resampled.index=resampled.index.droplevel(0)

#detrended = detrend(df,method='difference')

#f = fourier(df,[10,15],method='difference')

##results = wadl(df,[15])
##line = results.wadl[15]
# m = momentum(df, [10])
# res = m.close[10]
# s = stochastic(df, [14, 15])
# res = s.close[14]
# w = williams(df, [15])
# res = w.close[15]
# p = proc(df, [30])
# res = p.proc[30]
# AD = adosc(df, [30])
# res = AD.AD[30]
# m = macd(df, [15, 30])
# res = m.signal
# c = cci(df, [30])
# res = c.cci[30]
# b = bollinger(df, [20], 2)
# res = b.bands[20]
# avs = paverage(df, [20])
# res = avs.avs[20]
s = slopes(df, [20])
res = s.slope[20]

# (3)Plot

trace0 = go.Ohlc(
    x=df.index,
    open=df.open,
    high=df.high,
    low=df.low,
    close=df.close,
    name="Currency Quote",
)
trace1 = go.Scatter(x=df.index, y=ma)
trace2 = go.Ohlc(x=HA.index,open=HA.open,high=HA.high,low=HA.low,close=df.close,name='Heiken Ashi')
##trace2 = go.Scatter(x=df.index, y=fourier)

data = [trace0, trace1, trace2]

fig = tools.make_subplots(rows=2, cols=1, shared_xaxes=True)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)

py.offline.plot(fig, filename="EurUsdPlotly.html")
