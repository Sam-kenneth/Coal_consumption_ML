from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
# Assuming the PredictPipeline and CustomData are in the correct relative path
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import pandas as pd
import sys

# Standard boilerplate for Flask app initialization
application = Flask(__name__)
app = application
CORS(app) # Enable CORS for all routes

@app.route('/')
def home_page():
    """Renders the main page. Assumes index.html exists."""
    # Since we don't have the new input form, we use the existing template call
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Handles form submission for prediction.
    Expects data via form fields (used for web application frontends).
    """
    if request.method == 'GET':
        # Simply show the input form/page
        return render_template('index.html')
    else:
        try:
            # --- COLLECTING DATA FOR NEW ENERGY MODEL ---
            data = CustomData(
                # date is read as a string
                date = request.form.get('date'),
                # All other features are read as floats
                NSR = float(request.form.get('NSR')),
                Receipt = float(request.form.get('Receipt')),
                MC = float(request.form.get('MC')),
                EC = float(request.form.get('EC')),
                EN = float(request.form.get('EN')),
                EH = float(request.form.get('EH'))
            )
        except Exception as e:
            # Handle potential errors if form data is missing or cannot be converted to float
            print(f"Error processing form data: {e}", file=sys.stderr)
            return render_template('index.html', results="Error: Invalid input data format.", prediction_date=None)


        # Convert the raw data into the necessary DataFrame structure
        pred_df: pd.DataFrame = data.get_data_as_dataframe()
        
        # Log the input data
        print("Input DataFrame for Prediction:")
        print(pred_df)

        # Run the prediction pipeline
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)
        
        # Extract the single prediction result and round it
        results = round(pred[0], 2)
        
        # Extract the date for display (using the DataFrame index, which is the date)
        prediction_date = pred_df.index[0].strftime('%Y-%m-%d')
        
        # Return the result back to the template
        return render_template('index.html', 
                               results=f"Predicted Consumption: {results}", 
                               prediction_date=prediction_date)
    
@app.route('/predictAPI',methods=['POST'])
def predict_api():
    """
    Handles API requests for prediction.
    Expects data as JSON payload (used for external services).
    """
    if request.method=='POST':
        try:
            # --- COLLECTING DATA FOR NEW ENERGY MODEL FROM JSON ---
            data = CustomData(
                date = request.json['date'],
                NSR = float(request.json['NSR']),
                Receipt = float(request.json['Receipt']),
                MC = float(request.json['MC']),
                EC = float(request.json['EC']),
                EN = float(request.json['EN']),
                EH = float(request.json['EH'])
            )
        except (KeyError, ValueError) as e:
            # Handle JSON parsing or data conversion errors
            return jsonify({'error': 'Invalid JSON format or missing key.', 'details': str(e)}), 400

        # Convert the raw data into the necessary DataFrame structure
        pred_df = data.get_data_as_dataframe()
        
        # Run the prediction pipeline
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)

        # Format the result for JSON output
        dct = {'predicted_consumption': round(pred[0], 2), 'date': data.date.strftime('%Y-%m-%d')}
        return jsonify(dct)

if __name__ == '__main__':
    # Running the Flask app
    app.run(host='0.0.0.0', port=8000, debug=True)