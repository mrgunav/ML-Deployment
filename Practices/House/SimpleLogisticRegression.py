from flask import Flask, jsonify, request, render_template
import pandas as pd
import os
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np
import codecs, json 

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html');


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.form.to_dict();
        to_predict_list=list(data.values())
        print(to_predict_list)
        logistic = joblib.load("./python_logistic_model.pkl")
        height = int(to_predict_list[0])
        weight =int(to_predict_list[1])
        Prediction = logistic.predict([[height,weight]])
        if Prediction[0] == 0:
            result = 'Male'
        else:
            result = 'Female'    
    return result;

if __name__ == '__main__':
    app.run(debug=True)