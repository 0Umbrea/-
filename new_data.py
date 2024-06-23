import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

file_path = "E:\model2\creditcard_resampled.csv"
df = pd.read_csv(file_path)

print(df.info())
print(df.isnull().sum())

def plot_distribution(data, feature, target):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x=feature, hue=target, multiple="stack", kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

features = df.columns.drop('Class')

plt.figure(figsize=(18, 14))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

for feature in features:
    plot_distribution(df, feature, 'Class')

