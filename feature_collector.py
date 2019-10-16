import pandas as pd
import numpy as numpy
from feature_functions import *

# Load our CSV Data

data = pd.read_csv("Data/EurUsdHour.csv")
data.columns = ["Date", "open", "high", "low", "close", "AskVol"]
data = data.set_index(pd.to_datetime(data.Date))
data = data.drop("Date", 1)
data.columns = ["open", "high", "low", "close", "AskVol"]
prices = data.drop_duplicates(keep=False)

# Create Lists for Each Period requires by our functions

momentumKey = [3, 4, 5, 8, 9, 10]
stochasticKey = [3, 4, 5, 8, 9, 10]
williamsKey = [6, 7, 8, 9, 10]
procKey = [12, 13, 14, 15]
wadlKey = [15]
adoscKey = [2, 3, 4, 5]
macdKey = [15, 30]
cciKey = [15]
bollingerKey = [15]
heikenashiKey = [15]
paverageKey = [2]
slopeKey = [3, 4, 5, 10, 20, 30]
fourierKey = [10, 20, 30]
sineKey = [5, 6]

keylist = [
    momentumKey,
    stochasticKey,
    williamsKey,
    procKey,
    wadlKey,
    adoscKey,
    macdKey,
    cciKey,
    bollingerKey,
    heikenashiKey,
    paverageKey,
    slopeKey,
    fourierKey,
    sineKey,
]

# Calculate all the features

momentumDict = momentum(prices, momentumKey)
print("1")
stochasticDict = stochastic(prices, stochasticKey)
print("2")
williamsDict = williams(prices, williamsKey)
print("3")
procDict = proc(prices, procKey)
print("4")
wadlDict = wadl(prices, wadlKey)
print("5")
adoscDict = adosc(prices, adoscKey)
print("6")
macdDict = macd(prices, macdKey)
print("7")
cciDict = cci(prices, cciKey)
print("8")
bollingerDict = bollinger(prices, bollingerKey, 2)
print("9")

hkaprices = prices.copy()
hkaprices["Symbol"] = "SYMB"
HKA = OHLCresample(hkaprices, "15H")
heikenDict = heikenashi(HKA, heikenashiKey)
print("10")

paverageDict = paverage(prices, paverageKey)
print("11")
slopeDict = slopes(prices, slopeKey)
print("12")
fourierDict = fourier(prices, fourierKey)
print("13")
sineDict = sine(prices, sineKey)
print("14")

# Create list of dictionaries

dictlist = [
    momentumDict.close,
    stochasticDict.close,
    williamsDict.close,
    procDict.proc,
    wadlDict.wadl,
    adoscDict.AD,
    macdDict.line,
    cciDict.cci,
    bollingerDict.bands,
    heikenDict.candles,
    paverageDict.avs,
    slopeDict.slope,
    fourierDict.coeffs,
    sineDict.coeffs,
]

# List of 'base' column names:
colFeat = [
    "momentum",
    "stoch",
    "will",
    "proc",
    "wadl",
    "adosc",
    "macd",
    "cci",
    "bollinger",
    "heiken",
    "paverage",
    "slope",
    "fourier",
    "sine",
]

# Populate the Masterframe

masterFrame = pd.DataFrame(index=prices.index)

for i in range(0, len(dictlist)):
    if colFeat[i] == "macd":
        colID = colFeat[i] + str(keylist[6][0]) + str(keylist[6][1])
        masterFrame[colID] = dictlist[i]
    else:
        for j in keylist[i]:
            for k in list(dictlist[i][j]):
                colID = colFeat[i] + str(j) + str(k)
                masterFrame[colID] = dictlist[i][j][k]

threshold = round(0.7 * len(masterFrame))

masterFrame[["open", "high", "low", "close"]] = prices[["open", "high", "low", "close"]]

# Heikenashi is resampled ==> empty data in between

masterFrame.heiken15open = masterFrame.heiken15open.fillna(method="bfill")
masterFrame.heiken15high = masterFrame.heiken15high.fillna(method="bfill")
masterFrame.heiken15low = masterFrame.heiken15low.fillna(method="bfill")
masterFrame.heiken15close = masterFrame.heiken15close.fillna(method="bfill")

# Drop columns with 30% or more NAN Data

masterFrameCleaned = masterFrame.copy()
masterFrameCleaned = masterFrameCleaned.dropna(axis=1, thresh=threshold)
masterFrameCleaned = masterFrameCleaned.dropna(axis=0)

masterFrameCleaned.to_csv("Data/masterFrame.csv")
print("Completed feature calculations")

