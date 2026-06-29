import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("insurance.csv")
df = pd.get_dummies(df)
df = df.astype(float)

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1)

])
model.compile(optimizer="adam",loss="mse")
X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)
model.fit(X_train,y_train, epochs=100)

pred = model.predict(X_test)
mse=mean_squared_error(pred,y_test)
r2 = r2_score(pred,y_test)
print(f"mse: {mse}")
print(f"r2: {r2}")
