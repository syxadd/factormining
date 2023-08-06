#### Pytorch factor mining
[Export StockData]
#### 导出数据到每个个体，先打开该文件修改路径
python scripts/prepare_trainData.py

## 旧的处理方法在这
# python scripts/export_data_tosubs220820.py

# python scripts/export_data_tosubs.py

# python scripts/export_data_tosubs220820.py



## Train the network
[train command]
python trainMultifactor.py -opt configs/alphanetv2.yml

python trainMultifactor.py -opt configs/alphanetv3.yml

python trainMultifactor.py -opt configs/alphanetv2mod.yml

python trainMultifactor.py -opt configs/alphanetv3mod.yml

python trainMultifactor.py -opt configs/factornet4enco220728.yml


[test command]
#python predict.py

#python predictMulti.py

python predictMulti.py -opt configs/alphanetv3mod.yml -checkpoint D:/syx-working/quant/220727code-alphanetv3mod/experiments/20220727-11-38-47_Alphanetv3GRU/train_state_epoch25.tar 




[Plot results]
#export PYTHONPATH=PYTHONPATH:./
python scripts/plot_predictions.py


## 数据来自
https://cn.investing.com/rates-bonds/china-1-year-bond-yield-historical-data
