import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("insurance.csv")
df = pd.get_dummies(df)

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

X = df.drop("charges",axis=1)
y = df["charges"]

rf = Pipeline([("scalar",StandardScaler()),("model",RandomForestRegressor())])
xg = Pipeline([("scalar",StandardScaler()),("model",XGBRegressor())])
lr = Pipeline([("scalar",StandardScaler()),("model",LinearRegression())])

for name,model in[("rf",rf),("xg",xg),("lr",lr)]:
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    print(f"MSE: {mean_squared_error(pred,y_test)}")
    print(f"score: {r2_score(pred,y_test)}")