import os
from sklearn.metrics import classification_report,  roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import xgboost as xgb 
import joblib
import matplotlib.pyplot as plt
df = pd.read_csv("../data_science/earthquakes_filtred.csv")

good = np.load('good_idx.npy')
bad = np.load('bad_idx.npy')
diff = np.load('../data_science/difficult_negative_idx.npy')

subset3 = df.loc[diff]
subset1 = df.loc[good]
subset2 = df.loc[bad]

final_df = pd.concat([subset1, subset3, subset2])
print(final_df.columns)
print(len(final_df))
X = final_df.drop(columns=["properties.tsunami"]).values
y = final_df["properties.tsunami"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf_rf = RandomForestClassifier(n_estimators=200)
clf_rf.fit(X_train, y_train)

y_pred = clf_rf.predict(X_test)
y_proba = clf_rf.predict_proba(X_test)[:, 1]
rf_metrics = classification_report(y_test, y_pred, output_dict=True)
auc = roc_auc_score(y_test, y_proba)
rf_metrics['AUC'] = auc
print(rf_metrics)

clf_lr = LogisticRegression(max_iter=1000)
clf_lr.fit(X_train, y_train)

y_pred = clf_lr.predict(X_test)
y_proba = clf_lr.predict_proba(X_test)[:, 1]
LR_metrics = classification_report(y_test, y_pred, output_dict=True)
auc = roc_auc_score(y_test, y_proba)
LR_metrics['AUC'] = auc
print(LR_metrics)

clf_xgb = xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss')
clf_xgb.fit(X_train, y_train)

y_pred_xgb = clf_xgb.predict(X_test)
y_proba_xgb = clf_xgb.predict_proba(X_test)[:, 1]
xgb_metrics = classification_report(y_test, y_pred_xgb, output_dict=True)
auc = roc_auc_score(y_test, y_proba_xgb)
xgb_metrics['AUC'] = auc
print(classification_report(y_test, y_pred_xgb))


accuracies = [LR_metrics['accuracy'], rf_metrics['accuracy'], xgb_metrics['accuracy']]
f1_scores = [
    LR_metrics['macro avg']['f1-score'], 
    rf_metrics['macro avg']['f1-score'], 
    xgb_metrics['macro avg']['f1-score']
]
auc_scores = [LR_metrics['AUC'], rf_metrics['AUC'], xgb_metrics['AUC']]

max_acc_idx = np.argmax(accuracies)
max_f1_idx = np.argmax(f1_scores)
max_auc_idx = np.argmax(auc_scores)

print(f"Acurácia máxima atingida: {accuracies[max_acc_idx]}")
print(f"F1-Score máximo atingido: {f1_scores[max_f1_idx]}")
print(f"AUC máxima: {auc_scores[max_auc_idx]}")

models = {0:'Logistic Regression', 1: 'Random Forest', 2:'XGradient Boost'}

print(f"\n Modelo de melhor Acurácia: {models[max_acc_idx]}")
print(f"\n Modelo de melhor F1-score: {models[max_f1_idx]}")
print(f"\n Modelo de melhor AUC: {models[max_auc_idx]}")

os.makedirs('../model', exist_ok=True)
joblib.dump(scaler, '../model/scaler.joblib')
#joblib.dump(clf_lr, '../model/best_model.joblib')

if models[max_acc_idx] == 'Logistic Regression':
    joblib.dump(clf_lr, '../model/best_model.joblib')
elif models[max_acc_idx] == 'Random Forest':
    joblib.dump(clf_rf, '../model/best_model.joblib')
elif models[max_acc_idx] == 'XGradient Boost':
    joblib.dump(clf_xgb, '../model/best_model.joblib')    

        
model_names = list(models.values()) 

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Comparação de Métricas dos Modelos', fontsize=16)


colors = ['#1f77b4', '#ff7f0e', '#2ca02c']


ax1.bar(model_names, accuracies, color=colors)
ax1.set_title('Acurácia')
ax1.set_ylabel('Score')
ax1.set_ylim(0, max(accuracies) * 1.1) 


ax2.bar(model_names, f1_scores, color=colors)
ax2.set_title('F1-Score (Macro Avg)')
ax2.set_ylabel('Score')
ax2.set_ylim(0, max(f1_scores) * 1.1)

ax3.bar(model_names, auc_scores, color=colors)
ax3.set_title('AUC')
ax3.set_ylabel('Score')
ax3.set_ylim(0, max(auc_scores) * 1.1)


plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 


plt.savefig('../comparacao_modelos.png')

# Exibe o gráfico
plt.show()