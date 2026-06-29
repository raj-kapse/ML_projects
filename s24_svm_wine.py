from sklearn.svm import SVC
from sklearn.datasets import load_wine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier()
lr = LogisticRegression()

wine = load_wine()

X = wine.data
y = wine.target

sv = Pipeline([("scalar",StandardScaler()),("model", SVC())])
lr = Pipeline([("scaler ", StandardScaler()),("model", LogisticRegression())])

for name, model in [("random forest", rf),("logistic regression", lr),("svc",sv)]:
    scores = cross_val_score(model,X,y,cv=5)
    print(f"scores:{np.mean(scores):.4f}, (+/- {np.std(scores):.4f})")

