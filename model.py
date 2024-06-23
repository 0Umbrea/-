import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import MiniBatchKMeans
from xgboost import XGBClassifier

# 定义目标变量名称并替换特殊字符
target_name = "Class"
replace_str = [':', '[', ']', '（', '）', '！', '＠', '＃', '￥', '％', '…', '《', '》', '【', '】', ' ']
for s in replace_str:
    target_name = target_name.replace(s, '_')

# 读取训练集和验证集数据
train, validation = pd.read_csv("E:\\model2\\信用卡诈骗3_train.csv"), pd.read_csv("E:\\model2\\信用卡诈骗3_validation.csv")
validation = validation[train.columns]

# 分割特征和目标变量
X_train, y_train = train.drop(columns=target_name), train[target_name]
X_val, y_val = validation.drop(columns=target_name), validation[target_name]

# 使用MiniBatchKMeans进行数据采样
n_clusters = 1000
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, max_iter=100, n_init=10)
kmeans.fit(X_train)

# 选择每个聚类中的一部分样本
sampled_indices = []
for cluster in range(n_clusters):
    cluster_indices = np.where(kmeans.labels_ == cluster)[0]
    sampled_indices.extend(np.random.choice(cluster_indices, size=len(cluster_indices)//10, replace=False))

X_train = X_train.iloc[sampled_indices]
y_train = y_train.iloc[sampled_indices]

# 定义模型及其超参数
models = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000),
        'params': {
            'C': [0.01, 0.1, 0.25]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(),
        'params': {
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.5, 0.7, 1.0],
            'colsample_bytree': [0.5, 0.7, 1.0]
        }
    }
}

# 进行网格搜索和模型评估
for name, model in models.items():
    grid_search = GridSearchCV(model['model'], model['params'], cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    models[name]['best_model'] = grid_search.best_estimator_
    y_pred = grid_search.predict(X_val)
    scores = cross_val_score(model['best_model'], X_train, y_train, cv=5, scoring='f1')
    print(f"模型: {name}")
    print("最好的参数:", grid_search.best_params_)
    print("混淆矩阵:\n", confusion_matrix(y_val, y_pred))
    print("分类报告:\n", classification_report(y_val, y_pred))
    print("交叉验证f1:", scores)
    print("平均f1得分:", np.mean(scores))
    cv_results = cross_val_score(model['best_model'], X_train, y_train, cv=5, scoring='accuracy')
    print(f'交叉验证得分: {cv_results.mean():.4f} ± {cv_results.std():.4f}')

# 绘制所有模型的ROC曲线
plt.figure(figsize=(10, 8))
for name, model in models.items():
    y_pred_proba = model['best_model'].predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc='lower right')
plt.show()