from flask import Flask,render_template,request
import sklearn
import numpy as np
import pandas as pd
import pickle
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

@app.route('/')
@app.route('/index')

def index():
	return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
	if request.method == 'POST':
		data = request.form["data"]
		x = list(map(float,data.split(', ')))
		predictions = []
		for t in range(10):
			model = ARIMA(x, order=(5, 1, 0))
			model = model.fit()
			output = model.forecast()
			pred = output[0]
			predictions.append(pred)
			x.append(pred)
		return render_template("index.html", b = predictions)

if __name__ == '__main__':
	app.debug = True
	app.run(host = "0.0.0.0", port = 5000)