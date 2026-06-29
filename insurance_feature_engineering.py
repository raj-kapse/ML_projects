import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("insurance.csv")
df = pd.get_dummies(df)
df["age_bmi"] = df["age"]*df["bmi"]
df["age_smoker"] = df["age"] * df["smoker_yes"]

print(df.head())
print(df.skew())
print(df.info())
print(df.describe())
print(df.groupby("smoker_yes")["charges"].mean())

model = Pipeline([("scalar",StandardScaler()),("model",RandomForestRegressor())])

X = df.drop("charges", axis=1)
y = np.log1p(df["charges"])

X_train , X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
model.fit(X_train,y_train)

pred = np.expm1(model.predict(X_test))
y_test_actual = np.expm1(y_test)

print(f"score: {r2_score(y_test_actual,pred)}")