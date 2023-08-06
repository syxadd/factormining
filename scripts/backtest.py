import os
from cv2 import dft
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
根据预测结果进行回测;

每十天进行交易匹配
"""
## 测试开始到结束前的时间日期
start_date = "2021-09-01"
end_date = "2022-02-01"

extend_end_date = "2022-03-01"  # 对交易日期进行扩展，使得预测的区间能覆盖里面

## 预测结果文件
prediction_file = r"/home/ubuntu/syx/quant/220913newInterval/test_results/20220919-00-08-55_FactorNet-LargerLevelLearnable-30-10pred-220917-100pred/test_results.npz"
# time_length = 30
# 预测未来交易日的长度(这里是预测未来第几天)
pred_length = 10


## 指数代码权重文件（文件夹，里面包含了一共日期下的所有数据信息）
indexcodes_first_file = "../stockdata/index_components/000300.SH/bench_stocks_weights_20201231_000300.SH_citics1.csv"
# indexcodes_first_file = "../stockdata/index_components/000905.SH/bench_stocks_weights_20201231_000905.SH_citics1.csv"
# indexcodes_first_file = "../stockdata/index_components/000852.SH/bench_stocks_weights_20201231_000852.SH_citics1.csv"

indexcodes_folder = "../stockdata/index_components/000300.SH"  # 沪深300
# indexcodes_folder = "../stockdata/index_components/000905.SH"  # 中证500
# indexcodes_folder = "../stockdata/index_components/000852.SH"  # 中证1000








## 每次调仓的日期 - 根据预测结果自动调整，这里不再提前调整；
# adjust_datelist = []

## 指数权重变化的日期
change_dates = []

total_change_portion = 0.5

increase_point = 1 # 以百分比为单位


######## 投资成本
## 初始资金投入
initial_invest = 1e6

## 交易成本
# cost_ratio = 0.002 # 调仓时基于资产价值的千分之二收费
cost_ratio = 0.0 # 调仓时不收费

# 无风险利率：选取年化收益，直接取了2021年的平均
Rf = 2.34



## 输出文件夹
out_dir = "backtest_results/"
# labelname = "hushen300-0cost"
# labelname = "hushen300-0.002cost-debug"
labelname = "index-0.002cost-debug"
# labelname = "zhongzheng500-0cost"
# labelname = "zhongzheng500-0.002cost"
# labelname = "zhongzheng1000-0cost"
# labelname = "zhongzheng1000-0.002cost"



######## 分隔线
#
######## 以下是测试用的代码
## 先把预测结果构建数据库
def build_database(predictions: dict, start_date = None, end_date = None):
	"""
	利用预测结果，构建数据表。
	"""
	stock_names = predictions['stock_names']
	results = predictions['results']
	total_length = len(stock_names)
	## build dataframe
	dataframes = []
	for i in range(total_length):
		stock_name = stock_names[i]
		result = results[i]

		df_prediction = pd.DataFrame(data=result)
		df_prediction['df_date'] = pd.to_datetime(df_prediction['df_date'])

		if start_date:
			loc = (df_prediction['df_date'] >= start_date ) & (df_prediction['df_date'] < end_date)
			df_prediction = df_prediction[loc]

		df_prediction['stock_name'] = stock_name

		## location
		dataframes.append(df_prediction)

	df_data = pd.concat(dataframes, axis=0)
	df_data.sort_values(by='df_date', inplace=True)

	return df_data


def scan_indexfiles(folder: str, start_date=None, end_date=None):
	"""
	扫描文件夹下所有日期的指数权重数据并进行日期筛选
	"""
	namelist = [name for name in os.listdir(folder) if name.endswith(".csv")]
	datelist = []

	for name in namelist:
		# 比如文件名为："bench_stocks_weights_20220907_000852.SH_citics1"
		datestr = name[21:29]
		temp = list(datestr)
		temp.insert(4, '-') # "2022-0907"
		temp.insert(7, '-') # "2022-09-07"
		datestr = "".join(temp)
		datelist.append(datestr)

	df_indexfiles = pd.DataFrame(
		data=dict(date=datelist, name=namelist)
	)

	df_indexfiles['date'] = pd.to_datetime(df_indexfiles['date'])
	df_indexfiles.sort_values(by="date", inplace=True)

	if start_date:
		df_indexfiles = df_indexfiles[(df_indexfiles['date'] >= start_date) & (df_indexfiles['date'] < end_date)]
		
	df_indexfiles.reset_index(inplace=True)
	return df_indexfiles


# def selection_stock(df_predictions: pd.DataFrame, step_days: int, selection_num: int):
# 	"""
# 	Selection for predictions.
# 	"""

# 	## predictions has 3 columns: "df_date", "predict", "gt_return10"
# 	selections = []
	
# 	dates = df_predictions['df_date'].drop_duplicates()
# 	datelist = [dates[i] for i in range(0, len(dates), step_days)]
# 	for each_time in datelist:
# 		df_eachtime = df_predictions[df_predictions['df_date'] == each_time]

# 		sorted_data = df_eachtime.sort_values(by='predict', ascending=False)
# 		df_kselection = sorted_data.iloc[:selection_num]
# 		selections.append({
# 			"date": each_time,
# 			"df_selection": df_kselection,
# 		})

	# return selections

def get_predictions_index(df_indexcodes: pd.DataFrame, df_predictions: pd.DataFrame, prediction_day: int):
	"""
	根据给定的指数股票代码组合，从结果中找出对未来的时间预测。
	"""
	pass



# def analyze_selections(selections: list, trade_cost_ratio: float, initial_money = 10000):
# 	"""
# 	Analyze selection results of stocks.
# 	'selections' contains:
# 		[{
# 			"date": datetime64,  "df_selection": pd.DataFrame,
# 		}]
#
# 	直接假定交易按成交额千分之五收费
# 	"""
# 	length = len(selections)
# 	test_results = []
#
# 	money = initial_money
# 	assets = []
#
# 	for i, selection_each in enumerate(selections):
# 		df_kselection = selection_each['df_selection']
# 		weights = np.ones_like(df_kselection['df_date'])
# 		weights = weights / weights.sum()
# 		real_return = df_kselection['gt_return10'] * weights
#
# 		cost = real_return.abs()

		

def get_stockindex_weights(df_predict: pd.DataFrame, df_indexcodes: pd.DataFrame, predict_date, change_portion: float, increase_point: float):
	"""
	利用预测结果进行指数增强。根据
	必须包含预测当天的结果和调仓那天的权重。
	预测和股票代码只能包含那一天的结果，不能含两天以上。
		df_predict: 来自预测结果构造的DataFrame，包含四列；
		df_indexcodes: 来自wind终端获得的指数代码；
		change_portion: 单边的增强change_portion的范围比率。涨的加上跌的总共为2*change_portion;
	返回指数成分代码和对应新的权重。
	"""
	df_pred = df_predict

	newdf_codeweights = df_indexcodes.copy(deep=True)


	## 筛选出当天的行业权重
	s_info_citics_weights = df_indexcodes[['s_info_citics1_name', 'industry_weight']].drop_duplicates()
	# print(s_info_citics_weights)

	## 筛选股票列表的预测
	s_info_windcode = df_indexcodes['s_info_windcode']
	subdf_pred = df_pred[df_pred['stock_name'].isin(s_info_windcode)]

	# industry_length = 0
	industry_length = len(s_info_citics_weights)
	for industry_index in range(industry_length):
		# 逐个筛选行业股票
		industry_name = s_info_citics_weights.iloc[industry_index][0]
		subdf_indexcodes =  df_indexcodes[df_indexcodes['s_info_citics1_name'] == industry_name]
		# print(f"Industry: {industry_name}, length : {len(subdf_indexcodes)};")
		# sub_windcode = subdf_indexcodes['s_info_windcode']
		df_industrypred = subdf_pred[subdf_pred['stock_name'].isin(subdf_indexcodes['s_info_windcode'])]

		# 筛选日期
		# current_day = np.datetime64("2021-03-01")
		if isinstance(predict_date, str):
			predict_date = np.datetime64(predict_date)

		# subloc = (df_industrypred['df_date'] >= predict_date) & (df_industrypred['df_date'] < predict_date+1) # not supported by pandas timestamp
		subloc = df_industrypred['df_date'] == predict_date
		df_industrypred = df_industrypred[subloc]
		## 股票排序
		df_industrypred = df_industrypred.sort_values(by="predict", ascending=False)

		## 根据排名进行选股
		# 这里采用行业中性化处理：行业的权重与指数的权重不变，然后行业内部进行投资组合优化
		# 对前15%表现好的增加 1% 的权重，对后 15% 表现差的 降低 1% 的权重；increase_point为权重变化，change_portion为调整比例；
		# portion = 0.15
		portion = change_portion
		#### 平衡买入和卖出的比例
		total = len(df_industrypred)
		left = int(portion*total)
		right = int((1-portion)*total)
		while total - right > left:
			right += 1
		while left > total - right:
			left -= 1
		
		## 增加比例
		increase_ratio = increase_point
		# increase_ratio = 0.5
		increase_codes = df_industrypred.iloc[:left]["stock_name"]
		## 减少比例
		# 貌似减少的比例还不能低于0，只能等于0，因此增加的比例部分不能超过减少的比例部分，选取增加中从后往前数多少个减少相应的比例。
		decrease_codes = df_industrypred.iloc[right:]['stock_name']
		

		# 划分比例
		# 先各自增加删除 increase_ratio
		subloc = newdf_codeweights['s_info_windcode'].isin(increase_codes)
		# subdf = 
		newdf_codeweights.loc[subloc, 'i_weight'] += increase_ratio
		# newdf_codeweights[subloc]['i_weight'] += increase_ratio  ## setting with copy
		subloc = newdf_codeweights['s_info_windcode'].isin(decrease_codes)
		# subdf = newdf_codeweights[subloc]
		newdf_codeweights.loc[subloc, 'i_weight'] -= increase_ratio
		# newdf_codeweights[subloc]['i_weight'] -= increase_ratio

		# 然后再对小于0的权重进行填补
		except_loc = newdf_codeweights[subloc]['i_weight'] < 0
		# 记录小于0的权重
		decrease_ratio = newdf_codeweights[subloc]['i_weight'][except_loc].to_numpy().copy()
		# 惨痛教训，pandas这里引用只能引用一次才能赋值
		newdf_codeweights.loc[subloc & except_loc, 'i_weight'] = 0
		# newdf_codeweights[subloc]['i_weight'][except_loc] = 0
		length = int(except_loc.sum())
		# 再去掉靠后的几个增加的codes
		increase_codes = increase_codes.iloc[::-1].iloc[:length]
		# 进行增删
		subloc = newdf_codeweights['s_info_windcode'].isin(increase_codes)
		# subdf = newdf_codeweights[newdf_codeweights['s_info_windcode'] in increase_codes]
		newdf_codeweights.loc[subloc, 'i_weight'] += decrease_ratio # here 'decrease_ratio' < 0
		# newdf_codeweights.loc[subloc, 'i_weight'] = newdf_codeweights[subloc]['i_weight'].to_numpy() + decrease_ratio # here 'decrease_ratio' < 0
		## 现在应该得到了权重了。
		newdf_codeweights

		# 行业权重不变
		# subdf_loc = df_indexcodes['s_info_citics1_name'] == industry_name
		# newdf_codeweights.loc[subdf_loc, 'industry_weight']
		# newdf_codeweights.loc[subdf_loc, 'i_weight'].sum()

	# 返回新的代码和对应的权重
	return newdf_codeweights

def compute_index_price(df_indexweights: pd.DataFrame):
	"""
	计算指数组合的价格，包含两列，权重 "i_weight" 和单价。
	"""
	singleprice = df_indexweights['i_weight'] * df_indexweights['s_dq_close'] / 100 # 去掉权重百分比
	return singleprice.sum()

def compute_amount_withprice(df_indexweights: pd.DataFrame, investMoney: float):
	"""
	根据价格和权重计算购买到每个股票的数量。
	"""
	portfolios = df_indexweights['i_weight'] * investMoney / 100
	amounts = portfolios / df_indexweights['s_dq_close']
	return amounts

def compute_difference_weights(df_oldweights: pd.DataFrame, df_newweights: pd.DataFrame):
	"""
	比较新的权重和旧的权重之间的差异，然后进行权重替换。
	权重的变化在 "change_weight" 里面。以百分数为单位。
	"""
	columns = ['s_info_windcode', 'i_weight']
	# df1 = pd.DataFrame(data=dict(s_info_windcode=df_oldweights['s_info_windcode'], old_weight=df_oldweights['i_weight']), index=df_oldweights.index)
	df2 = pd.DataFrame(data=dict(s_info_windcode=df_newweights['s_info_windcode'], new_weight=df_newweights['i_weight']), index=df_newweights.index)
	df_diff = pd.merge(df_oldweights[columns], df2, how="outer", on="s_info_windcode")
	df_diff = df_diff.fillna(0)
	df_diff['change_weight'] = df_diff['new_weight'] - df_diff['i_weight']

	return df_diff



def compute_sharp_ratio(returns:pd.Series, Rf):
	"""
	计算Sharp Ratio
	"""
	sigma = np.std(returns)
	means = np.mean(returns) - np.mean(Rf)
	
	return means / sigma

def analyze_trade(df_investweights: pd.DataFrame, df_trade_data: pd.DataFrame):
	"""
	回测分析投资的表现，基于历史数据分析。
	
	"""


	pass








def main():
	######## 回测收益的主体代码
	## 读取数据信息	 - 指数代码数据库
	in_folder = indexcodes_folder
	# filelist = [name for name in os.listdir(in_folder) if name.endswith(".csv")]
	# filelist.sort()
	# filelist
	print("Scan all index files...")
	df_indexfiles = scan_indexfiles(in_folder, start_date, end_date)
	change_dates = df_indexfiles['date']


	#### 读取交易日期
	print("Read trade date series...")
	trade_dts = pd.read_csv("data/trade_dt.csv").trade_dt
	trade_dts = pd.to_datetime(trade_dts)



	# 构建预测结果数据库
	print("Get predictions...")
	predictions = np.load(prediction_file, allow_pickle=True)
	stock_codes = predictions['stock_names']
	# 选取整个包含预测范围的时间段构建数据库  范围：[start_date, end_date)
	df_pred = build_database(predictions, start_date=start_date, end_date=extend_end_date)
	# del(predictions)



	## 构造交易日期序列
	dateseries = np.arange(start_date, extend_end_date, dtype="datetime64[D]")
	trade_dates = dateseries[np.isin(dateseries, trade_dts)]
	trade_dates = pd.to_datetime(trade_dates)
	# 获取调仓日期（根据预测时间来定）
	adjust_datelist = [trade_dates[i] for i in range(0, len(trade_dates), pred_length)]
	# 这里提取[start_date, end_date) 作为每日观测的区间
	watch_dates = trade_dates[(trade_dates >= start_date) & (trade_dates < end_date)]

	new_weights_list = []
	origin_weights_list = []


	## 这里添加初始的状态的indexcodes
	df_initial_indexcodes = pd.read_csv(indexcodes_first_file, encoding='gbk')
	df_today_indexcodes = df_initial_indexcodes.copy()
	df_newcodeweights = df_initial_indexcodes.copy()
	

	print("Start computing weights on each day.")



	## 开始每天的投资计算
	for i, current_day in enumerate(watch_dates):
		# 先获得当天的index股票权重
		if current_day in change_dates:
			# 月底index权重会变化（当前有每月的权重）
			# df_today_indexcodes
			name = df_indexfiles.loc[df_indexfiles['date']==current_day, 'name']
			df_today_indexcodes = pd.read_csv(os.path.join(in_folder, name))
		else:
			pass

		# 然后看看是否是预测并调仓的那天
		if current_day in adjust_datelist:
			## 调仓那天确定新的权重
			predict_date = trade_dates[i + pred_length]
			df_pred_day = df_pred[df_pred['df_date'] == predict_date]
			## 获得新的权重调整
			df_newcodeweights = get_stockindex_weights(df_pred_day, df_today_indexcodes, predict_date=predict_date, change_portion=total_change_portion/2, increase_point=increase_point)

			new_weights_list.append(df_newcodeweights)
			origin_weights_list.append(df_today_indexcodes)
			msg = f"Date: {current_day}, change portfolio."

		else:
			# 非调仓就保留原有的权重
			origin_weights_list.append(df_today_indexcodes)
			new_weights_list.append(df_newcodeweights)
			msg = f"Date: {current_day}, retain the previous state."


		## Part: 对比当天指数的变化和 自选投资组合的变化，这个见后面。

		### 输出信息
		print(msg)

	
	#### 初始价格的计算
	index_asset_eachday = initial_invest
	invest_asset_eachday = initial_invest
	index_assets = []
	invest_assets = []
	# close_price_list = []

	
	index_amount_list = []
	invest_amount_list = []
	initial_status = True

	## 计算指数价格变化？
	print("Compute price trends...")
	index_price_list = []
	invest_price_list = []
	# 
	#### 读取 完整交易数据
	print("Reading all trade data...")
	df_trade_data = pd.read_csv("../quant-data/status_data_all.csv", encoding='gbk')
	df_trade_data['trade_dt'] = pd.to_datetime(df_trade_data['trade_dt'].apply(str))
	df_trade_data = df_trade_data[(df_trade_data['trade_dt'] >=  start_date) & (df_trade_data['trade_dt'] <= end_date) ]
	## NOTE: 进行复权价格调整
	df_trade_data['s_dq_close'] = df_trade_data['s_dq_close'] * df_trade_data['s_dq_adjfactor']

	## 对每天的交易信息进行表格进行合并？
	for i, current_date in enumerate(watch_dates):
		## 整理记录每次的当期价格
		df_trade_daydata = df_trade_data[df_trade_data['trade_dt'] == current_date]
		trade_columns = ["s_info_windcode", "s_dq_close", "s_dq_open"]

		## 合并股票和交易数据
		# 合并指数投资数据
		# 先筛选出含indexcodes的数据
		index_codes = origin_weights_list[i]['s_info_windcode']
		subdf_trade_daydata = df_trade_daydata[df_trade_daydata['s_info_windcode'].isin(index_codes)]
		origin_weights_list[i] = pd.merge(origin_weights_list[i], subdf_trade_daydata[trade_columns], on="s_info_windcode")
		# 合并新的投资组合数据
		index_codes = new_weights_list[i]['s_info_windcode']
		subdf_trade_daydata = df_trade_daydata[df_trade_daydata['s_info_windcode'].isin(index_codes)]
		new_weights_list[i] = pd.merge(new_weights_list[i], subdf_trade_daydata[trade_columns], on="s_info_windcode")


		## 统计每天收盘时的资产数量，以及资产总量的变动--根据价格和资产配置数量计算总量
		## 初始的资产数量配置
		if initial_status:
			index_amount = compute_amount_withprice(origin_weights_list[i], index_asset_eachday)
			invest_amount = compute_amount_withprice(new_weights_list[i], invest_asset_eachday)
			initial_status = False

		## 份额变化时，计算新的份额。并计算调仓的成本
		# 指数权重变化时，计算指数配额变化
		if not initial_status and current_day in change_dates:
			# 计算配额变化带来的成本
			df_diff = compute_difference_weights(origin_weights_list[i-1], origin_weights_list[i])
			#交易成本按照第二天的vwap算；
			change_cost = df_diff['change_weight'].abs().sum() * cost_ratio * index_asset_eachday
			index_asset_eachday = index_asset_eachday - change_cost / (1 - cost_ratio)

			# 计算去掉交易成本后的资产数量和资产变化
			index_amount = compute_amount_withprice(origin_weights_list[i], index_asset_eachday)

		# 投资组合权重变化时，计算组合的配比变化
		if not initial_status and current_day in adjust_datelist:
			# 计算配额变化带来的成本
			df_diff = compute_difference_weights(new_weights_list[i-1], new_weights_list[i])
			#交易成本按照第二天的vwap算；
			change_cost = df_diff['change_weight'].abs().sum() * cost_ratio * invest_asset_eachday
			invest_asset_eachday = invest_asset_eachday - change_cost / (1 - cost_ratio)

			## 计算去掉交易成本后的资产数量和资产变化
			invest_amount = compute_amount_withprice(new_weights_list[i], invest_asset_eachday)


		## 计算变动后的交易成本（变动费用）
		# 成本计算方式为 Vold = (Vold - Vnew)*cost_ratio + Cost + Vnew  --> Vnew = Vold - Cost / (1-cost_ratio)
		# index_asset_eachday = index_asset_eachday - index_cost / (1 - cost_ratio)
		# invest_asset_eachday = invest_asset_eachday - invest_cost / (1 - cost_ratio)


		## 记录资产数量配置
		index_amount_list.append(index_amount)
		invest_amount_list.append(invest_amount)

		# 每一期收盘之后的资产变化
		index_asset_eachday = (index_amount * origin_weights_list[i]['s_dq_close']).sum()
		invest_asset_eachday = (invest_amount * new_weights_list[i]['s_dq_close']).sum()

		## 记录每天收盘时的资产变化
		index_assets.append(index_asset_eachday)
		invest_assets.append(invest_asset_eachday)



		## 计算每一期的价格
		# index_price_list.append(compute_index_price(origin_weights_list[i]))
		# invest_price_list.append(compute_index_price(new_weights_list[i]))

		print(f"Date: {current_date}, index price {index_asset_eachday} ; invest price {invest_asset_eachday} .")


	## 准备输出结果
	out_folder = os.path.join(out_dir, labelname)
	if not os.path.exists(out_folder):
		os.makedirs(out_folder)
	
	## 画图
	plt.plot(watch_dates, index_assets, label="Index Investment Asset")
	plt.plot(watch_dates, invest_assets, label="Enhanced Investment Asset")
	plt.title("Asset trend.")
	plt.legend()
	png_file = os.path.join(out_folder, "compare_assets.png")
	plt.savefig(png_file)
	plt.close()
	print("Save figure to file.")

	## 保存结果，分析和记录当天的结果
	# 计算每日收益率
	index_assets = pd.Series(index_assets)
	index_returns = index_assets / index_assets.shift(1) - 1
	invest_assets = pd.Series(invest_assets)
	invest_returns = invest_assets / invest_assets.shift(1) - 1

	# 绘制收益率的图
	plt.plot(watch_dates, index_returns, label="Index Investment")
	plt.plot(watch_dates, invest_returns, label="Enhanced Investment")
	plt.title("Daily Return")
	plt.legend()
	png_file = os.path.join(out_folder, "compare_returns_daily.png")
	plt.savefig(png_file)
	plt.close()

	# 计算每次调仓时（预测的天数间隔）间隔收益率
	index_assets_predlen = index_assets[::pred_length]
	invest_assets_predlen = invest_assets[::pred_length]
	watch_dates_predlen = watch_dates[::pred_length]
	index_returns_predlen = index_assets_predlen / index_assets_predlen.shift(1) - 1
	invest_returns_predlen = invest_assets_predlen / invest_assets_predlen.shift(1) - 1
	plt.plot(watch_dates_predlen, index_returns_predlen, label="Index Investment")
	plt.plot(watch_dates_predlen, invest_returns_predlen, label="Enhanced Investment")
	plt.title("Predict Length Return")
	plt.legend()
	png_file = os.path.join(out_folder, "compare_return_predday.png")
	plt.savefig(png_file)
	plt.close()

	## 分析结果
	# 比较年化的结果
	index_sharpRatio = compute_sharp_ratio(index_returns * 365, Rf)
	invest_sharpRatio = compute_sharp_ratio(invest_returns * 365, Rf)

	msg = f"Index Sharp Ratio: {index_sharpRatio} ; Invest Sharp Ratio: {invest_sharpRatio} ."

	## 计算与最初的收益差距
	index_totalreturn = index_assets.iloc[-1] / initial_invest - 1
	invest_totalreturn = invest_assets.iloc[-1] / initial_invest - 1
	msg += "\n" + f"Index 总共收益率：{index_totalreturn * 100}%, Invest 总共收益：{invest_totalreturn * 100}% , 后者减前者：{(invest_totalreturn - index_totalreturn)*100}% ."

	# 计算收益率波动性
	index_dailysigma = np.std(index_returns)
	invest_dailysigma = np.std(invest_returns)
	msg += "\n" + f"Index daily sigma: {index_dailysigma} ; Invest daily sigma: {invest_dailysigma} ."


	print(msg)
	with open(os.path.join(out_folder, "output.txt"), 'w') as f:
		f.write(msg)

	## 结束
	print("End simulation test.")









def test_code():
	## 测试代码用的函数
	raise NotImplementedError

	# 信息
	in_folder = "../stockdata/index_components/000300.SH"
	filelist = [name for name in os.listdir(in_folder) if name.endswith(".csv")]
	filelist.sort()
	filelist


	# predict_file = "data/test_results_30-10pred_100ep.npz"
	# predictions = np.load(predict_file, allow_pickle=True)
	# predictions

	## select datetime
	# start_date = "2021-01-01"
	# end_date = "2022-03-01"
	# df_pred = build_database(predictions, start_date=start_date, end_date=end_date)
	# df_pred

	index = 162
	df_indexcodes = pd.read_csv(os.path.join(in_folder, filelist[index]), encoding='gbk')
	print(filelist[index])
	print(df_indexcodes)

	current_day = "20210301"
	# df_pred_day = df_pred[df_pred['df_date'] == current_day]
	df_pred_day = pd.read_csv("data/df_pred_day.csv")
	df_pred_day['df_date'] = pd.to_datetime(df_pred_day['df_date'])
	print(df_pred_day)

	newdf_codeweights = get_stockindex_weights(df_pred_day, df_indexcodes, pred_day=10, increase_point=0.5, change_portion=0.15)
	newdf_codeweights

	diff = newdf_codeweights['i_weight'] - df_indexcodes['i_weight']
	print(diff.abs().sum())

	print("OK")

	## 结束
	print("End simulation test.")









if __name__ == "__main__":
	main()

	# test_code()

	print("End of the process.")

