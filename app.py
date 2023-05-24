from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import pickle
from statsmodels.tsa.arima.model import ARIMA
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go

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
		data = data[::-1]
		model = pickle.load(open('model_pickle_apple_arima.sav', 'rb'))
		pred = model.predict(start = len(data), end = len(data)+days, typ = 'levels')
		x = [i for i in range(1, days+2)]
		fig1 = go.Figure([go.Scatter(x=data['Date'], y=data[' Close/Last'])])
		fig2 = go.Figure([go.Scatter(x=x, y=pred)]) 
		graphJSON1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
		graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
		return render_template('index.html', b = pred, graphJSON1=graphJSON1, graphJSON2=graphJSON2)

if __name__ == '__main__':
	app.debug = True
	app.run(host = "0.0.0.0", port = 5000)
