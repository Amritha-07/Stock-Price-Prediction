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
		days = request.form["data"]
		days = int(days) - 1
		data = pd.read_csv('HistoricalQuotes.csv', parse_dates = True)
		model = pickle.load(open('model_pickle_apple_arima.sav', 'rb'))
		pred = model.predict(start = len(data), end = len(data)+days, typ = 'levels')
		return render_template('index.html', b = pred)

if __name__ == '__main__':
	app.debug = True
	app.run(host = "0.0.0.0", port = 5000)