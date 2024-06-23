# 使用Featuretools进行特征工程
import featuretools as ft
import pandas as pd
import numpy as np
from featuretools.primitives import AggregationPrimitive, TransformPrimitive
from featuretools.variable_types import Numeric, Categorical, Boolean
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# 自定义聚合和转换操作
class AggMin(AggregationPrimitive):
    name = "agg_min"
    input_types = [Numeric]
    return_type = Numeric

    def get_function(self):
        return np.min

class AggMax(AggregationPrimitive):
    name = "agg_max"
    input_types = [Numeric]
    return_type = Numeric

    def get_function(self):
        return np.max

class AggMean(AggregationPrimitive):
    name = "agg_mean"
    input_types = [Numeric]
    return_type = Numeric

    def get_function(self):
        return np.mean

class Entropy(AggregationPrimitive):
    name = "entropy"
    input_types = [Categorical]
    return_type = Numeric

    def get_function(self):
        def entropy_func(series):
            value, counts = np.unique(series, return_counts=True)
            return entropy(counts, base=2)
        return entropy_func

class Combine(TransformPrimitive):
    name = "combine"
    input_types = [Categorical, Categorical]
    return_type = Categorical

    def get_function(self):
        return lambda x, y: x.astype(str) + y.astype(str)

class MinFunc(TransformPrimitive):
    name = "min_func"
    input_types = [Numeric, Numeric]
    return_type = Numeric

    def get_function(self):
        return np.minimum

class MaxFunc(TransformPrimitive):
    name = "max_func"
    input_types = [Numeric, Numeric]
    return_type = Numeric

    def get_function(self):
        return np.maximum

class Residual(TransformPrimitive):
    name = "residual"
    input_types = [Numeric]
    return_type = Numeric

    def get_function(self):
        return lambda x: x - x.astype(int)

class Add(TransformPrimitive):
    name = "add"
    input_types = [Numeric, Numeric]
    return_type = Numeric

    def get_function(self):
        return np.add

class Subtract(TransformPrimitive):
    name = "subtract"
    input_types = [Numeric, Numeric]
    return_type = Numeric

    def get_function(self):
        return np.subtract

class Multiply(TransformPrimitive):
    name = "multiply"
    input_types = [Numeric, Numeric]
    return_type = Numeric

    def get_function(self):
        return np.multiply

class Divide(TransformPrimitive):
    name = "divide"
    input_types = [Numeric, Numeric]
    return_type = Numeric

    def get_function(self):
        def safe_divide(x, y):
            return np.divide(x, y, out=np.zeros_like(x), where=y!=0)
        return safe_divide

class Square(TransformPrimitive):
    name = "square"
    input_types = [Numeric]
    return_type = Numeric

    def get_function(self):
        return np.square

# 读取数据并创建DataFrame
data = pd.read_csv(r'E:\\model2\\creditcard_resampled.csv')
df = pd.DataFrame(data)
target = df.pop('Class')

# 创建实体集并进行特征工程
es = ft.EntitySet(id="data")
es = es.entity_from_dataframe(entity_id="data", dataframe=df, index="index", make_index=False)
new_columns = []

feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity="data", max_depth=1, max_features=100,
                                      trans_primitives=[MinFunc, MaxFunc, Add, Multiply, Divide, Square])
for col in feature_matrix.columns:
    new_columns.append(feature_matrix[col])
final_df = pd.concat([df] + new_columns, axis=1)
final_df['Class'] = target
print(feature_matrix)

# 分割特征和目标变量
X = final_df.drop(columns=['Class'])
X = X.drop(columns=['index'])
y = final_df['Class']

# 训练随机森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 绘制特征重要性
importances = model.feature_importances_
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)
top_features = feature_importances.head(68)

plt.figure(figsize=(12, 8))
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top Feature Importances')
plt.gca().invert_yaxis()
plt.show()

# 选择重要特征进行进一步分析
selected_features = top_features['feature'].values
final_df_top = final_df[selected_features]