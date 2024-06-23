
# 信用卡欺诈检测

## 目录结构
|文件名称|文件介绍                                                   |
|----------------|----------------------------------------|
|creditcard4.csv|原始数据集
|creditcard_resampled.csv |进行过采样后产生的新数据集                       
|信用卡诈骗3_train.csv          |训练集                |
|信用卡诈骗3_validation.csv   |测试集|
|automl_record.csv|自动化机器学习得出的模型|
|features.csv|特征工程产生的新特征
|bestmodel.py|自动化机器学习产生的最佳模型|
|model.py|利用网格搜索进行探索的三种模型|
|params.py|进行自动化机器学习|
|smote.py|进行数据处理|
|features.py|进行特征工程|
|fake.ipynb|整合的所有代码，易于查看

## 安装依赖

请确保您的环境中安装了以下依赖库：
`pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost featuretools flaml`

## 运行

### 运行脚本

按照以下顺序运行脚本：
1.  数据预处理：`python model2/smote.py`
2.  模型训练：`python model2/model.py 以及 python model2/bestmodel.py(可见最终模型训练结果)`
3.  特征工程：`python model2/features.py(示例,最终选择的特征查看features.csv)`
4.  数据可视化：`python model2/data.py 以及 python model2/new_data.py(新旧数据比较)`
5.  自动化机器学习：`python model2/automl.py(示例,最终训练模型以及参数查看automl_record.csv)`

### 整体运行
运行fake.ipynb文件，或者打开查看所有运行结果
## 总结

本项目旨在建立一个高效的信用卡欺诈检测模型，通过数据预处理、模型训练与评估、特征工程和数据可视化等步骤，最终选择了性能最优的模型并进行了评估。自动化机器学习工具FLAML的使用简化了模型选择和超参数优化的过程。我们发现了一些对信用卡欺诈检测非常重要的特征，并通过特征工程和特征选择提高了模型的性能。
