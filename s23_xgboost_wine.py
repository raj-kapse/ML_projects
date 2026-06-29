from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_wine
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

wine = load_wine()
X = wine.data
y = wine.target

xgb = XGBClassifier()
rf = RandomForestClassifier()

lr = Pipeline([("scaler", StandardScaler()),("model", LogisticRegression(max_iter=1000))])

for name, model in [("Random Forest", rf), ("xgboost",xgb), ("Logistic Regression", lr)]:
    scores = cross_val_score(model,X,y,cv=5)
    print(f"{name}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")