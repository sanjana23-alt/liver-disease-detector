"""Reads the dataset, trains the model, and writes model.js for the website."""
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

COLS = ["Age", "Gender", "TB", "DB", "Alkphos", "Sgpt", "Sgot", "TP", "ALB", "AG", "Label"]

df = pd.read_csv("indian_liver_patient.csv")
df.columns = COLS
df["AG"] = df["AG"].fillna(df["AG"].median())
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Label"] = df["Label"].map({1: 1, 2: 0})  # 1 = liver disease

X, y = df.drop("Label", axis=1), df["Label"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler().fit(X_tr)
model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
model.fit(scaler.transform(X_tr), y_tr)

pred = model.predict(scaler.transform(X_te))
prob = model.predict_proba(scaler.transform(X_te))[:, 1]
metrics = {
    "accuracy": round(accuracy_score(y_te, pred), 3),
    "recall": round(recall_score(y_te, pred), 3),
    "precision": round(precision_score(y_te, pred), 3),
    "auc": round(roc_auc_score(y_te, prob), 3),
    "train": len(X_tr), "test": len(X_te),
}

out = {
    "features": list(X.columns),
    "mean": [round(v, 4) for v in scaler.mean_],
    "scale": [round(v, 4) for v in scaler.scale_],
    "coef": [round(v, 4) for v in model.coef_[0]],
    "intercept": round(float(model.intercept_[0]), 4),
    "median": [round(float(v), 2) for v in X.median()],
    "metrics": metrics,
}
with open("model.js", "w") as f:
    f.write("const MODEL = " + json.dumps(out, indent=2) + ";\n")
print("model.js written", metrics)
