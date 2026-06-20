import pandas as pd
from sklearn.linear_model import Ridge,Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("dataset.csv")
df = df.select_dtypes(include="number")

X = df.drop("Aggregate rating",axis=1)
y= df["Aggregate rating"] 

ridge = Pipeline([("scalar",StandardScaler()),("model",Ridge())])
lasso = Pipeline([("scalar",StandardScaler()),("model",Lasso(alpha=0.01))])


for name,model in [("ridge",ridge),("lasso",lasso)]:
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=45)
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    
    print(f"MSE: {mean_squared_error(y_test,pred)}")
    print(f"R2: {r2_score(y_test, pred)}")
    
print(lasso.named_steps["model"].coef_)