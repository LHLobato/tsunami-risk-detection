from sklearn.metrics import classification_report,  roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import xgboost as xgb 

df = pd.read_csv("earthquakes_filtred.csv")

good = np.load('good_idx.npy')
bad = np.load('bad_idx.npy')

subset1 = df.loc[good]
subset2 = df.loc[bad]

final_df = pd.concat([subset1, subset2])
print(len(final_df))
X = final_df.drop(columns=["properties.tsunami"]).values
y = final_df["properties.tsunami"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
clf = RandomForestClassifier(n_estimators=200)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print("AUC: ",  roc_auc_score(y_test, y_proba)) 

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print("AUC: ",  roc_auc_score(y_test, y_proba))

clf_xgb = xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss')
clf_xgb.fit(X_train, y_train)

y_pred_xgb = clf_xgb.predict(X_test)
y_proba_xgb = clf_xgb.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred_xgb))
print(f"AUC (XGB): {roc_auc_score(y_test, y_proba_xgb)}\n")