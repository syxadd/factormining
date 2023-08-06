import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.metrics import evaluate_single

"""
Plot the predictions in an interval. 
Also evaluate the results in the interval.
"""

predict_file = "/home/ubuntu/syx/quant/220913newInterval/test_results/20220915-11-49-59_FactorNet-ID18-LargerLevelLearnable-30train15pred-100epoch/test_results.npz"

start_date = "20210101"
end_date = "20220401"


title_name = "2021-2022 Year Prediction"
out_folder = "plot_results/learnable-30train-15pred-100epoch/Predict2022"

## 按照4:1的训练和测试集划分，必须要至少有1430天的数据，才能保证最后一年在测试集中。
minlength = 2250


## evaluate
# updownthreshold=0.005
updownthreshold=0.0


#### ***** for old 220917before split usage 
# 如果split了就不按照minlength来划分。
split = True
split_ratio = [4, 1]
train_split, test_split = split_ratio
train_split = train_split / sum(split_ratio)
test_split = test_split / sum(split_ratio)
#### ---- 

def main():
	print(f"""
	Prediction file : {predict_file}
	start to end : {start_date} -- {end_date}
	Name : {title_name}
	Output folder : {out_folder}
	""")
	if not os.path.exists(out_folder):
		os.makedirs(out_folder, exist_ok=True)

	predictions = np.load(predict_file, allow_pickle=True)
	stock_names = predictions['stock_names']
	results = predictions['results']
	total_nums = len(stock_names)

	print("Begin plotting...")

	## for save usage
	# ndays = []
	df_evalresults = pd.DataFrame()


	for index in tqdm(range(total_nums)):
		stockname = stock_names[index]
		## 忽略含有st的股票
		# if stockname.startswith(("ST", "*ST")):
		# 	print(f"Ignore stock : {stockname}")
		# 	continue


		df_results = pd.DataFrame(data=results[index])
		# predicts = results[index]['predict']
		# gt_return10 = results[index]['gt_return10']
		# df_date = results[index]['df_date']

		# df_date = df_results['df_date']
		# df_date = pd.Series(df_date)
		# df_date = pd.to_datetime(df_date.apply(str))
		df_results['df_date'] = pd.to_datetime(pd.Series(df_results['df_date']))
		df_results.sort_values(by='df_date', inplace=True)
		df_results.reset_index(inplace=True)

		## split data in test set or the interval
		if split:
			## split the test set ratio
			df_results = df_results.iloc[int(train_split*len(df_results)):]
		else:
			if len(df_results) < minlength:
				continue
		
		df_date = df_results['df_date']
		predicts = df_results['predict']
		gt_return10 = df_results['gt_return10']

		
		# if len(df_date) < minlength:
		# 	continue

		
		## 提取日期
		sub_loc = (df_date >= start_date) & (df_date < end_date)
		sub_date = df_date[sub_loc]
		sub_pred = predicts[sub_loc]
		sub_gt = gt_return10[sub_loc]

		if len(sub_date) <= 100:
			# 最少得有100天的数据吧
			continue


		## evaluation
		eval_result = evaluate_single(sub_pred, sub_gt, updownthreshold=updownthreshold)
		eval_result['name'] = stock_names[index]
		eval_result['ndays'] = len(sub_date)
		df_evalresults = df_evalresults.append(eval_result, ignore_index=True)



		## plot data
		plt.hlines(0, xmin=sub_date.iloc[0], xmax=sub_date.iloc[-1], colors="orange", linestyles="dashed")
		p1, = plt.plot(sub_date, sub_gt, color='green', label="sub_gt")
		p2, = plt.plot(sub_date, sub_pred, color='red', label="sub_pred")
		# plt.legend([None, 'Real Return', 'Prediction'])
		plt.legend([p1, p2], ['Return', 'Prediction'])
		plt.xlabel("Date")
		plt.ylabel("Return rate")
		plt.title(f"{title_name} : {stock_names[index]}.")

		png_file = os.path.join(out_folder, stock_names[index] + ".png")
		plt.savefig(png_file)
		plt.close()

	## save tables
	df_evalresults.to_csv(os.path.join(out_folder, "avg_metrics.csv"))

	print("End of the process.")





if __name__ == "__main__":
	main()