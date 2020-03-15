from flask import Flask, jsonify, request, render_template
import pandas as pd
import os
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
import numpy as np
import codecs, json 

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    print("I am hre in predict function")
    Prediction = 1
    if request.method == 'POST':
        try:
            #data = request.get_json()
            data = request.form.to_dict()
            print(data)
            years_of_experience = float(data["yearsOfExperience"])
            print(years_of_experience)
            lin_reg = joblib.load("./linear_regression_model.pkl")
            print(lin_reg.intercept_)
        except ValueError:
            return jsonify("Please enter a number.")
        print("Prediction = ",lin_reg.predict([[years_of_experience]]))
        Prediction = lin_reg.predict([[years_of_experience]])
    print(type(Prediction))
    a = Prediction.tolist()
    print(a)
    json_dump = json.dumps(a)
    print(json_dump)
    #return (json_dump)

    return render_template("result.html",a = a)


@app.route("/retrain", methods=['POST'])
def retrain():
    if request.method == 'POST':
        data = request.get_json()

        try:
            training_set = joblib.load("./training_data.pkl")
            training_labels = joblib.load("./training_labels.pkl")

            df = pd.read_json(data)

            df_training_set = df.drop(["Salary"], axis=1)
            df_training_labels = df["Salary"]

            df_training_set = pd.concat([training_set, df_training_set])
            df_training_labels = pd.concat([training_labels, df_training_labels])

            new_lin_reg = LinearRegression()
            new_lin_reg.fit(df_training_set, df_training_labels)

            os.remove("./linear_regression_model.pkl")
            os.remove("./training_data.pkl")
            os.remove("./training_labels.pkl")

            joblib.dump(new_lin_reg, "linear_regression_model.pkl")
            joblib.dump(df_training_set, "training_data.pkl")
            joblib.dump(df_training_labels, "training_labels.pkl")

            lin_reg = joblib.load("./linear_regression_model.pkl")
        except ValueError as e:
            return jsonify("Error when retraining - {}".format(e))
        print("Retrained model successfully.")
        return jsonify(data = "Retrained model successfully.")


@app.route("/currentDetails", methods=['GET'])
def current_details():
    if request.method == 'GET':
        try:
            lr = joblib.load("./linear_regression_model.pkl")
            training_set = joblib.load("./training_data.pkl")
            labels = joblib.load("./training_labels.pkl")

            return jsonify({"score": lr.score(training_set, labels),
                            "coefficients": lr.coef_.tolist(), "intercepts": lr.intercept_})
        except (ValueError, TypeError) as e:
            return jsonify("Error when getting details - {}".format(e))


if __name__ == '__main__':
    app.run(port=9080, debug=True)
