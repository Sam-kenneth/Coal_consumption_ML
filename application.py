from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import pandas as pd
import sys


application = Flask(__name__)
app = application
CORS(app) 

@app.route('/')
def home_page():
    
    
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    
    if request.method == 'GET':
        
        return render_template('index.html')
    else:
        try:
            
            data = CustomData(
                
                date = request.form.get('date'),
                
                NSR = float(request.form.get('NSR')),
                Receipt = float(request.form.get('Receipt')),
                MC = float(request.form.get('MC')),
                
                EN = float(request.form.get('EN')),
                EH = float(request.form.get('EH'))
            )
        except Exception as e:
            
            print(f"Error processing form data: {e}", file=sys.stderr)
            return render_template('index.html', results="Error: Invalid input data format.", prediction_date=None)


        
        pred_df: pd.DataFrame = data.get_data_as_dataframe()
        
        
        print("Input DataFrame for Prediction:")
        print(pred_df)

        
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)
        
        
        results = round(pred[0], 2)
        
        
        prediction_date = pred_df.index[0].strftime('%Y-%m-%d')
        
        
        return render_template('index.html', 
                               results=f"Predicted Consumption: {results}", 
                               prediction_date=prediction_date)
    
@app.route('/predictAPI',methods=['POST'])
def predict_api():
    
    if request.method=='POST':
        try:
            
            data = CustomData(
                date = request.json['date'],
                NSR = float(request.json['NSR']),
                Receipt = float(request.json['Receipt']),
                MC = float(request.json['MC']),
                
                EN = float(request.json['EN']),
                EH = float(request.json['EH'])
            )
        except (KeyError, ValueError) as e:
            
            return jsonify({'error': 'Invalid JSON format or missing key.', 'details': str(e)}), 400

        
        pred_df = data.get_data_as_dataframe()
        
        
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)

        
        dct = {'predicted_consumption': round(pred[0], 2), 'date': data.date.strftime('%Y-%m-%d')}
        return jsonify(dct)

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=8000, debug=True)