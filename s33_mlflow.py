from sklearn.datasets import load_iris
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import mlflow

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

model = RandomForestClassifier()

X = df.drop("target", axis=1)
y = df["target"]

with mlflow.start_run():
    mlflow.log_param("n_estimators",500)
    mlflow.log_param("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "Random_forest_iris")
    pred = model.predict(X_test)
    mlflow.log_metric("accuracy",accuracy_score(y_test, pred))

    print(f"accuracy score: {accuracy_score(y_test, pred)} and \n classification : {classification_report(y_test, pred)}")



