import pandas as pd
import numpy as np

"""
时间序列操作；假定
序列是时间正向变化的，从旧到新变化；

"""
## 1 波动性指标
## AD(集散量)
def ts_AD(high: pd.Series, low: pd.Series, close_price: pd.Series, volume: pd.Series, epsilon=1e-7):
	x = (close_price - low) - (high - close_price)
	x = x / (high - low + epsilon) * volume
	return x.cumsum()
	
# ATR 真实波幅  参考设置时间为14天：https://baike.baidu.com/item/%E7%9C%9F%E5%AE%9E%E6%B3%A2%E5%B9%85/175870
def ts_ATR(high: pd.Series, low: pd.Series, close_price: pd.Series, days: int = 14):
	close_1 = close_price.shift(1) # close_{t-1}
	TR = np.max([np.abs(high - low), np.abs(close_1 - high), np.abs(close_1 - low)], axis=0)
	ATR = pd.Series(TR, index=high.index).rolling(days).mean()
	return ATR


## ADTM(动态买卖人气指标)
def ts_ADTM(high: pd.Series, low:pd.Series, open_price: pd.Series, days_n: int, epsilon = 1e-7):
	open_price_1 = open_price.shift(1)
	loc = open_price <= open_price_1
	DTM = 0 * loc + np.max([high - open_price, open_price - open_price_1], axis=0) * (1-loc)
	loc = open_price < open_price_1
	DBM = (open_price - low) * loc

	STM = pd.Series(DTM).rolling(days_n).sum()
	SBM = pd.Series(DBM).rolling(days_n).sum()
	
	ADTM = (STM - SBM) / (STM + epsilon) * (STM > SBM) + (STM - SBM) / (SBM+epsilon) * (STM < SBM) 
	return ADTM

## CCI(商品通道指标)
def ts_CCI(high: pd.Series, low: pd.Series, close_price: pd.Series, days: int, epsilon=1e-7):
	TP = (high + low + close_price) / 3
	MA = close_price.rolling(days).sum() / days
	MD = (MA - close_price).rolling(days).sum()
	CCI = (TP - MA) / (MD+epsilon) * 0.015
	return CCI

## PSY(心理线)
def ts_PSY(open_price: pd.Series, close_price, days: int):
	# NOTE: will create NA values
	upcount = (close_price > open_price).rolling(days).sum()
	PSY = upcount / days * 100
	PSYMA = PSY.rolling(days).mean()
	return PSYMA

## RSI(相对强弱指标)
def ts_RSI(close_price: pd.Series, days: int, epsilon = 1e-7):
	price_diff = close_price - close_price.shift(1)
	A = ((price_diff > 0) * price_diff).rolling(days).sum()
	B = ((-1) * (price_diff < 0) * price_diff).rolling(days).sum()
	RSI = A / (A+B+epsilon)
	return RSI
	
## BIAS(乖离率)
def ts_BIAS(close_price: pd.Series, days: int, percent = False):
	avg_price = close_price.rolling(days).mean()
	BIAS = (close_price - avg_price) / avg_price 
	if percent :
		return BIAS * 100
	else: 
		return BIAS

## W%R(威廉指标)
def ts_WandR(high: pd.Series, low: pd.Series, close_price: pd.Series, days: int, percent = False, epsilon=1e-7):
	nday_high = high.rolling(days).max()
	nday_low = low.rolling(days).min()
	WR = (close_price- nday_low) / (nday_high - nday_low + epsilon)
	if percent :
		return WR * 100
	else:
		return WR

## KDJ(随机指标) --百度搜的
def ts_KDJ(high: pd.Series, low: pd.Series, close_price: pd.Series, day_n: int, epsilon=1e-7):
	high_max = high.rolling(day_n).max()
	low_min = low.rolling(day_n).min()
	RSV = (close_price - low_min) / (high_max - low_min + epsilon)

	ones = np.ones_like(high)
	K = ones * 50
	D = ones * 50
	# J = ones * 50
	for i in range(1, len(high)):
		if pd.isna(RSV[i]):
			continue
		K[i] = 2/3 * K[i-1] + 1/3 * RSV[i]
		D[i] = 2/3 * D[i-1] + 1/3 * K[i]
	J = 3*D - 2*K
	return K, D, J

## ASI(振动升降指标)
def ts_ASI(high: pd.Series, low: pd.Series, open_price: pd.Series, close_price: pd.Series, days: int):
	pre_close = close_price.shift(1)
	A = (high - pre_close).abs()
	B = (low - pre_close).abs()
	pre_low = low.shift(1)
	C = (high - pre_low).abs()
	D = (pre_close - open_price)
	E = (close_price - pre_close).abs()
	F = (close_price - open_price).abs()
	G = (close_price - open_price).shift(1).abs()
	
	R = (A + B/2 + D/4) * ((A >= B) & (A >= C)) + (B + A/2 + D/4) * ((B > A) & (B >= C)) + (C + D/4) * ((C > A) & (C > B))
	X = E + F/2 + G
	K = pd.Series(np.max([A, B], 0))
	SI = 16 + X/R * K
	ASI = SI.cumsum()
	return ASI

## OBV(能量潮)
def ts_OBV(volume: pd.Series, close_price: pd.Series):
	close_pre = close_price.shift(1)
	symbol = (close_price > close_pre) * 1 + (close_price < close_pre) * (-1)
	symVolume = volume * symbol
	return symVolume.cumsum()

## BRAR(人气意愿指标)
def ts_BRAR(high: pd.Series, low: pd.Series, open_price: pd.Series, close_price: pd.Series, days: int, epsilon=1e-8):
	"""
	Compute AR and BR.
	Return: 
		(AR, BR) two Series.
	"""
	AR = (high - open_price).rolling(days).sum() / (open_price - low + epsilon).rolling(days).sum()
	preclose = close_price.shift(1)
	BR = (high - preclose).rolling(days).sum() / (preclose - low + epsilon).rolling(days).sum()
	return AR, BR

## CR(能量指标)
def ts_CR(high: pd.Series, low: pd.Series, open_price: pd.Series, close_price: pd.Series, days: int, epsilon=1e-8):
	MID = (high + low + close_price) / 3
	CR = (high - MID.shift(1)).rolling(days).sum() / (MID.shift(1) - low + epsilon).rolling(days).sum() * 100
	return CR


## MFI(货币流量指数) / 资金流量指标
# MFI 尽量计算14天内的
def ts_MFI(high: pd.Series, low: pd.Series, close_price: pd.Series, volume: pd.Series, days: int, epsilon=1e-8):
	TP = (high + low + close_price) / 3
	MF = TP * volume.rolling(days).sum()
	preMF = MF.shift(1)
	PMF = (MF > preMF) * MF
	PMF = PMF.rolling(days).sum()
	NMF = (MF < preMF) * MF
	NMF = NMF.rolling(days).sum()
	MFI = 100 - (100 / (1 + PMF / (NMF + epsilon)))
	return MFI

## MA(移动平均线)
def ts_MA(close_price: pd.Series, days: int):
	return close_price.rolling(days).mean()

## MACD(平滑异同移动平均线)
# EMA计算参考这个：https://baike.baidu.com/item/EMA/12646151
def ts_MACD(close_price: pd.Series, days_n: int, days_m: int):
	# use for-loop 
	assert days_n > 0 and days_m > 0
	EMA_n = close_price.copy()
	EMA_m = close_price.copy()
	for i in range(1, len(EMA_n)):
		EMA_n[i] = 2 * EMA_n[i] / (days_n+1) + (days_n-1)/(days_n+1) * EMA_n[i-1]
		EMA_m[i] = 2 * EMA_m[i] / (days_m+1) + (days_m-1)/(days_m+1) * EMA_n[i-1]

	diff = EMA_n - EMA_m
	DEA = pd.Series(np.empty_like(diff.to_numpy(), dtype=np.float))
	DEA[0] = 0.2 * diff[0]
	for i in range(1, len(diff)):
		DEA[i] = 0.2 * diff[i] + 0.8 * DEA[i-1]

	return DEA

# EXPMA(指数平滑移动平均线)
def ts_EXPMA(close_price: pd.Series, nday: int):
	expma = pd.Series(np.empty_like(close_price.to_numpy(), dtype=np.float))
	expma[0] = close_price[0] / nday
	for i in range(1, len(expma)):
		expma[i] = (close_price[i] - expma[i-1]) / nday + expma[i-1]
	
	return expma
	

# BOLL 布林带
def ts_BOLL(close_price: pd.Series, nday: int):
	MA = close_price.rolling(nday).mean()
	pre_close = close_price.shift(1)
	MD = pd.Series(np.sqrt( ((pre_close - MA)**2).rolling(nday-1).sum() / nday ))
	MB = pre_close.rolling(nday-1).mean()
	UP = MB + MD * 2
	DOWN = MB - MD * 2
	return UP, MB, DOWN


















