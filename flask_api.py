import joblib
import numpy as np
from flask import Flask, request, jsonify

model = joblib.load("model.pkl")

app = Flask(__name__)

@app.route("/predict",methods = ["POST"])
def predict():
    data = request.json
    data = np.array(data)
    result = float(np.expm1(model.predict(data)[0]))

    return jsonify({"prediction": result})

if __name__ =="__main__":
    app.run(debug=True)