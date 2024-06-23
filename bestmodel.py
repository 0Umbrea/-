import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import lightgbm as lgbm

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)  # 计算混淆矩阵
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)  # 绘制热力图
    plt.xlabel('predict value')
    plt.ylabel('real value')
    plt.title('confusion_matrix')
    plt.show()

# 绘制ROC曲线
def plot_roc_curve():
    plt.figure(figsize=(10, 8))
    y_pred_proba = model.predict_proba(X_val)[:, 1]  # 预测概率
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)  # 计算假阳性率和真阳性率
    roc_auc = auc(fpr, tpr)  # 计算AUC
    plt.plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.2f})')  # 绘制ROC曲线
    plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate ')
    plt.ylabel('True Positive Rate')
    plt.title('ROC曲线')
    plt.legend(loc='lower right')
    plt.show()

# 评估模型得分
def eval_score(y_val, y_pred, y_proba=None):
    val_loss = log_loss(y_val, y_proba)  # 计算对数损失
    val_f1 = f1_score(y_val, y_pred, average='weighted')  # 计算F1得分
    val_accuracy = accuracy_score(y_val, y_pred)  # 计算准确率
    val_precision = precision_score(y_val, y_pred, average='weighted')  # 计算精确率
    val_recall = recall_score(y_val, y_pred, average='weighted')  # 计算召回率
    if y_proba.shape[1] > 1:
        num_classes = y_proba.shape[1]
        val_auc_roc = 0.0
        for class_idx in range(num_classes):
            val_auc_roc += roc_auc_score((y_val == class_idx).astype(int), y_proba[:, class_idx])
        val_auc_roc /= num_classes
    else:
        val_auc_roc = roc_auc_score(y_val, y_proba)  # 计算AUC得分
    return {
        "val_log_loss": val_loss,
        "val_f1": val_f1,
        "val_accuracy": val_accuracy,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_auc": val_auc_roc,
    }

target_name = "Class"
replace_str = [':', '[', ']', '（', '）', '！', '＠', '＃', '￥', '％', '…', '《', '》', '【', '】', ' ']
for s in target_name:
    if s in replace_str:
        target_name = target_name.replace(s, '_')

train, validation = pd.read_csv("E:\\model2\\信用卡诈骗3_train.csv"), pd.read_csv("E:\\model2\\信用卡诈骗3_validation.csv")
validation = validation[train.columns]

X_train, y_train = train.drop(columns=target_name), train[target_name]
X_val, y_val = validation.drop(columns=target_name), validation[target_name]

params = {
    "reg_alpha": 0.0009765625,
    "num_leaves": 743,
    "reg_lambda": 0.13107155388231867,
    "n_estimators": 549,
    "learning_rate": 0.34963748896822755,
    "colsample_bytree": 0.39701580008553367,
    "min_child_samples": 2
}

model = lgbm.LGBMClassifier(**params)  # 初始化LightGBM模型
model.fit(X_train, y_train)  # 训练模型

pred = model.predict(X_val)  # 预测验证集
proba = model.predict_proba(X_val)  # 预测概率

model_res = eval_score(y_val, pred, proba)  # 评估模型

for k, v in model_res.items():
    print(f"{k}: {v}")

labels = y_val.unique()
plot_confusion_matrix(y_val, pred, labels)  # 绘制混淆矩阵
plot_roc_curve()  # 绘制ROC曲线

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
print(f'交叉验证准确率: {cv_results.mean():.4f} ± {cv_results.std():.4f}')
print("分类报告:\n", classification_report(y_val, pred))  # 打印分类报告

scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
print("模型: lightgbm")
print("交叉验证F1:", scores)
print("平均F1:", np.mean(scores))
