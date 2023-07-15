from flask import Flask,request,app,jsonify,url_for,render_template
import json
import pickle
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    scaled_data = [float(x) for x in request.form.values()][:3] 
    unscaled_data = [float(x) for x in request.form.values()][3:]
    scaled_input = scalar.transform(np.array(scaled_data).reshape(1, -1))
    unscaled_data = np.array(unscaled_data).reshape(1, -1)
    final_input = np.concatenate((scaled_input, unscaled_data), axis=1)
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("index.html",prediction_text="YOUR PREMIUM PRICE IS {}".format(output))

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080)
