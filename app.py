from flask import Flask, render_template, request
import joblib
from datetime import datetime

app = Flask(__name__)

# Load trained model
model = joblib.load('model.pkl')


@app.route('/')
def home():
    return render_template('Home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'GET':
        return render_template('Application.html')

    try:
        data = [[
    float(request.form['CODE_GENDER']),
    float(request.form['FLAG_OWN_CAR']),
    float(request.form['FLAG_OWN_REALTY']),
    float(request.form['CNT_CHILDREN']),
    float(request.form['AMT_INCOME_TOTAL']),
    float(request.form['NAME_INCOME_TYPE']),
    float(request.form['NAME_EDUCATION_TYPE']),
    float(request.form['NAME_FAMILY_STATUS']),
    float(request.form['NAME_HOUSING_TYPE']),
    float(request.form['DAYS_BIRTH']),
    float(request.form['DAYS_EMPLOYED']),
    float(request.form['CNT_FAM_MEMBERS']),
    float(request.form['OCCUPATION_TYPE']),
    float(request.form['open_month']),
    float(request.form['end_months']),
    float(request.form['window'])
]]

        prediction = model.predict(data)[0]
        return render_template(
            'Result.html',
            prediction_output=prediction,
            used_algorithm="Random Forest Classifier",
            execution_timestamp=datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        )

    except Exception as e:
        return f"Error: {e}"


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)