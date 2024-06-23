from flaml import AutoML
import pandas as pd
import re

# 定义目标变量名称并替换特殊字符
target_name = "Class"
replace_str = [':', '[', ']', '（', '）', '！', '＠', '＃', '￥', '％', '…', '《', '》', '【', '】', ' ']
for s in replace_str:
    target_name = target_name.replace(s, '_')

# 读取训练集和验证集数据
train = pd.read_csv("E:\\model2\\信用卡诈骗3_train.csv")
validation = pd.read_csv("E:\\model2\\信用卡诈骗3_validation.csv")
validation = validation[train.columns]

# 分割特征和目标变量
X_train, y_train = train.drop(columns=target_name), train[target_name]
X_val, y_val = validation.drop(columns=target_name), validation[target_name]

# 使用FLAML进行自动化机器学习
automl = AutoML()
settings = {
    "time_budget": 600,  # 设置运行时间预算为600秒
    "metric": 'accuracy',  # 设置评估指标为准确率
    "task": 'classification',  # 设置任务类型为分类
    "log_file_name": 'automl.log',  # 设置日志文件名
    "estimator_list": ['lgbm', 'xgboost', 'rf', 'extra_tree'],  # 设置使用的模型列表
    "eval_method": 'auto',  # 设置评估方法为自动
    "n_splits": 20,  # 设置交叉验证的折数为20
}

# 训练自动化模型
automl.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **settings)
print("-----------------")

# 打印日志文件中前200个模型的参数
def process_estimators():
    try:
        for i in range(1, 200):
            print(automl.get_estimator_from_log("automl.log", i, "classification"))
    except ValueError:
        return 0

process_estimators()