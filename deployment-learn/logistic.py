from flask import Flask, jsonify, request, render_template;
import pandas as pd;
import numpy as np;
import codecs, json;
from sklearn.externals import joblib;

# APP INITILIZATION
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html");

@app.route('/predict', methods=['POST'])
def prediction():
    print("Prediction function started")
    if request.method == 'POST':
        # data = request.get_json()
        data = request.form.to_dict()
        data_val = list(data.values())
        data_val = list(map(int, data_val))
        print('data', data, data_val)
        log_reg = joblib.load("logistic_regression_model.pkl")
        # npArr = np.array(data_val).reshape(1,2)
        #print('npArr', npArr)
        y_pred = log_reg.predict([data_val])
        print('y_pred', y_pred[0])
    if y_pred[0] == 1:
        return 'Female'
    else:
        return 'Male'    
# Main Method
if __name__ == '__main__':
    app.run(debug=True)