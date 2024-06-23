# 读取数据进行进一步分析和可视化
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "E:\\creditcard4.csv"
df = pd.read_csv(file_path)

# 显示数据基本信息和缺失值统计
print(df.info())
print(df.isnull().sum())

# 绘制特征分布
def plot_distribution(data, feature, target):
    plt.figure(figsize=(6, 4))
    sns.histplot(data=data, x=feature, hue=target, kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

features = df.columns.drop('Class')

# 绘制相关矩阵
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.1f')
plt.title('Correlation Matrix')
plt.show()

# 绘制每个特征的分布图
class_data = df[df['Class'] == 1]
non_class_data = df[df['Class'] == 0]

for tmp in features:
    plot_distribution(df, tmp, "Class")