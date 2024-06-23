import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sweetviz

data = pd.read_csv("E:\creditcard4.csv")
X = data.drop(columns=['Class'])
y = data['Class']
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])
X['Time'] = scaler.fit_transform(X[['Time']])

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

data_resampled = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled, columns=['Class'])], axis=1)

data_resampled.to_csv(r"E:\model\creditcard_resampled.csv", index=False)

feature_file = pd.read_csv(r"E:\model2\features.csv")
automl_record = pd.read_csv(r"E:\model2\automl_record.csv")

plt.figure(figsize=(6, 3))
plt.table(cellText=feature_file.values, colLabels=feature_file.columns, loc='center')
plt.axis('off')
plt.show()
data = pd.read_csv(r"E:\creditcard4.csv")
new_data = pd.read_csv(r"E:\model2\creditcard_resampled.csv")
report = sweetviz.analyze(data)
report.show_notebook()

new_report = sweetviz.analyze(new_data)
new_report.show_notebook()
