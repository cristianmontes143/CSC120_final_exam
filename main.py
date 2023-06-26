from website import create_web, render_template
import pickle
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

web = create_web()

heart_disease_model = tf.keras.models.load_model('Neural_Network.h5')
# Load the saved logistic regression model using pickle
with open('heart_dsease.pickle', 'rb') as file:
    heart_disease_logistic = pickle.load(file)
with open('median_house.pickle', 'rb') as file:
    median = pickle.load(file)

@web.route('/heart-disease', methods=['GET', 'POST'])
def heart_disease():
    if request.method == 'POST':
# Load the saved neural network model
    # data = request.json

    
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trtbps = float(request.form['trtbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalachh = float(request.form['thalachh'])
        exng = float(request.form['exng'])
        oldpeak = float(request.form['oldpeak'])
        slp = float(request.form['slp'])
        caa = float(request.form['caa'])
        thall = float(request.form['thall'])
        
    
        # Preprocess the input values
        data = np.array([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]])
        output = heart_disease_model.predict(data)[0][0]
        
        # Return the prediction as a JSON response
        if output < 0.5:
            prediction = 'low risk of heart disease'
        else:
            prediction = 'high risk of heart disease'
            
    
        return render_template("result_heart_disease.html", result=prediction)
    
    # # Make the prediction using the loaded model
    # output = model.predict(input_values)[0][0]
    
    
    return render_template("Neural_network.html")



@web.route('/heart-disease-logistic', methods=['GET', 'POST'])
def heart_disease_logistic():
    if request.method == 'POST':
        
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trtbps = float(request.form['trtbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalachh = float(request.form['thalachh'])
        exng = float(request.form['exng'])
        oldpeak = float(request.form['oldpeak'])
        slp = float(request.form['slp'])
        caa = float(request.form['caa'])
        thall = float(request.form['thall'])

        # Preprocess the input values
        data = np.array([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]])
        output = heart_disease_logistic.predict(data)[0][0]
        
        # Return the prediction as a JSON response
        if output < 0.5:
            prediction = 'low risk of heart disease'
        else:
            prediction = 'high risk of heart disease'
            
    
        return render_template("result_heart_disease_logistic.html", result=prediction)
    
    return render_template("logistic_regression.html")


@web.route('/median_house', methods=['GET', 'POST'])
def median_house():
    
    if request.method == 'POST':
        
        interest_rate = float(request.form['interest_rate'])
        
        data = np.array([[interest_rate]])
        output = median.predict(data)[0]
        
        return render_template("result_linear.html", result=output)
        
    return render_template("linear_regression.html")
if __name__ == '__main__':
    
    web.run(debug=True)