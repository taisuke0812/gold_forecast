import pandas as pd
import numpy as np
import io
from google.colab import files
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
import warnings
import itertools
%matplotlib inline



#uploaded = files.upload()

#データを読み込む。このとき、日付がstr型だと都合が悪いのでPandas.Timestamp型に変換しておく
#データはここ(https://lets-gold.net/historical_data.php)から
#リンクから直接getできない様子なのでローカルに保存してから処理することにした 
#gold_data = pd.read_csv("historical_data_2016.csv",index_col = "DATE",parse_dates = ["DATE"])
gold_data = pd.read_csv("historical_data_2016.csv")
gold_data2 = pd.read_csv("historical_data_2017.csv")
#連続していない時系列のデータでは処理できない？
#まだよくわかっていないがすべて連続した日付と加工したところうまくいった
gold_data.index = pd.date_range("2016-01-01","2016-09-01",freq="D")
gold_data2.index = pd.date_range("2016-09-02","2017-05-06",freq="D")
del gold_data["DATE"]

#型を確認する
#print(type(gold_data.index))
#一応、各列の関係を調べておく
#print(gold_data.corr())
#今回のデータでは列間に大きな関係はなさそう
#必要のない列を削除しておく
gold_ = gold_data.drop(columns = ["PT_TOKYO","GOLD_NY","PT_NY","USDJPY"])
gold_.dropna()
gold2_ = gold_data2.drop(columns = ["DATE","PT_TOKYO","GOLD_NY","PT_NY","USDJPY"])

gold = pd.concat([gold_,gold2_])
gold.index = pd.date_range("2016-01-01","2017-05-06",freq = "D")

#まず、現状のデータを確認する。
#print(type(gold_["GOLD_TOKYO"]))
#plt.plot(gold_data["DATE"],gold_data["GOLD_TOKYO"])
#自己相関係数を計算する。
#gold_acf = sm.tsa.stattools.acf(gold_data["GOLD_TOKYO"], nlags=31)
#plt.plot(gold_acf)
#print(gold_acf)
#どうやら前日、前々日のデータがかなり影響を及ぼしているらしい

#まず、対数変換して変化をみてみる
#gold_data["GOLD_TOKYO"] = pd.Series(np.log(gold_data["GOLD_TOKYO"]))
#gold_acf = sm.tsa.stattools.acf(gold_data["GOLD_TOKYO"], nlags=31)
#print(gold_acf)
#どうやら自己相関係数は若干減少してしまう...
#gold_pacf = sm.tsa.stattools.pacf(gold_data["GOLD_TOKYO"], nlags=31, method='ols')
#print(gold_pacf)

# 偏自己相関係数のグラフを出力
#fig=plt.figure(figsize=(12, 8))
#ax = fig.add_subplot(212)
#自己相関係数
#sm.graphics.tsa.plot_acf(gold_data["GOLD_TOKYO"], lags=80,ax = ax) 
#偏自己相関係数
#fig = sm.graphics.tsa.plot_pacf(gold_data["GOLD_TOKYO"], lags=80, ax=ax)
#plt.show()
#2,7,46,69あたりが大きくなっている

#ADF検定を行う
#参照(https://blog.brains-tech.co.jp/entry/arima-tutorial-2)
#adf_result = sm.tsa.stattools.adfuller(gold_data["GOLD_TOKYO"],autolag='AIC')
#adf = pd.Series(adf_result[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
#print(adf)  


#print(adf_result)
#p値は0.64?
#階差をとってみる
#gold_diff = gold_.diff()
#gold_diff.index = gold_data.index
#gold_diff = gold_diff.dropna()
#print(gold_diff)
#plt.plot(gold_diff)
#plt.show()
#sm.graphics.tsa.plot_acf(gold_diff["GOLD_TOKYO"], lags=80,ax = ax) 

#sm.graphics.tsa.plot_pacf(gold_data["GOLD_TOKYO"], lags=80, ax=ax)

#selectparameter(gold_,10)
#(0, 1, 0), (0, 1, 1, 10)
#320点なら60が良い
#gold = gold[:320]
test = gold[:]
#N=320
#N=370
N =300
gold = gold[:N]
#グラフよりs=1,6,69あたりが良い？
ARIMA_gold = sm.tsa.statespace.SARIMAX(gold,order=(0, 1, 0),seasonal_order = (1,1,1,69), enforce_stationarity = False, enforce_invertibility = False,trend = "n").fit(trend='nc',disp=False)
print(ARIMA_gold.summary())
pred = ARIMA_gold.predict()
pred2 = ARIMA_gold.forecast(50)
#print(pred2)

#predict_dy = ARIMA_gold.get_prediction(start='2016-01-04', end='2016-12-30',dynamic='2016-12-22')
#print(pred,gold_diff)
#
#fig = plt.figure(figsize=(12,8))
#ax1 = fig.add_subplot(211)
#fig = sm.graphics.tsa.plot_acf(gold, lags=40, ax=ax1)
#ax2 = fig.add_subplot(212)
#fig = sm.graphics.tsa.plot_pacf(gold, lags=40, ax=ax2)


plt.plot(gold_,color = "b")
plt.plot(gold2_,color = "b")
plt.plot(pred,color = "r")
plt.plot(pred2,color="y")

plt.xlim(["2016-05-01","2017-06-06"])
plt.ylim([4400,5200])
plt.show()



data_score = 0
for i in range(49):
  if pred[i+1] - pred[i] > 0:
    if test.values[N+1+i] - test.values[N+i] > 0:
      data_score += 1
  if pred[i+1] - pred[i] < 0:
    if test.values[N+1+i] - test.values[N+i] < 0:
      data_score += 1

print("正解率:" + str(100*data_score/50) + "%")
#320:正答率38%
#370:正答率62%
#420:正答率42%

