import pandas as pd
import quandl as ql
import math, datetime
import numpy as np 
import imp
from sklearn import preprocessing, svm #svm - support vector machine, preprocessing helps with processing speeds, cross
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot");

ql.ApiConfig.api_key = "qmBXAzHoZuCBzvmrxUTs"

df = ql.get("WIKI/GOOGL") #df = data frame

df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]] #sets columns
df["highlowPercent"] = (df["Adj. High"] - df["Adj. Low"]) / df["Adj. Close"] * 100.0 #creating col
df["openclosePercent"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100.0 #creating col

df = df[["Adj. Close", "highlowPercent", "openclosePercent", "Adj. Volume"]] #resets columns

forcecastCol = "Adj. Close"
df.fillna(-99999, inplace=True) #used to make the data an outlier, instead of deleting the data  FILLNA fills all NaN values with zeroes 

forecastOut = int(math.ceil(0.01*len(df) - 5)) #used to predict 10 percent of the data frame
print(forecastOut)

df["Forecast"] = df[forcecastCol].shift(-forecastOut) #shifting the col negatively, therefore each row with be the adjust close 1% days into the future

x = np.array(df.drop(["Forecast"], 1)) #converted to a np array
x = preprocessing.scale(x) #scale with other values, but can add to processing time
xLately = x[-forecastOut:]
x = x[:-forecastOut]		 
df.dropna(inplace=True)

y = np.array(df["Forecast"])
y = np.array(df["Forecast"])

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = .2) #splits arrays into test  or train subsets

clf = LinearRegression(n_jobs = -1) #can use different algorithm, n_jobs = -1 will use all cores of cpu.     CLF Means classifier 
clf.fit(xTrain, yTrain) #goes with train
accuracy = clf.score(xTest, yTest) #goes with test
forecastSet = clf.predict(xLately)


print(forecastSet, accuracy, forecastOut)
df["Forecast"] = np.NaN

lastDate = df.iloc[-1].name #dont have date values
lastUnix = lastDate.timestamp()
oneDay = 86400
nextUnix = lastUnix + oneDay

for i in forecastSet:	#for forecast on the axis(graph)
	nextDate = datetime.datetime.fromtimestamp(nextUnix)
	nextUnix += oneDay
	df.loc[nextDate] = [np.NaN for x in range(len(df.columns) - 1)] + [i]  #list of values that are NaN. i is the forecast plus one value
#next date is the index
df["Adj. Close"].plot()
df["Forecast"].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show() 




