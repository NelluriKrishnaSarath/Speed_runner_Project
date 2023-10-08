# app.py

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model from the .pkl file
model = joblib.load('roadrunner_speed_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get user inputs from the form
            distance = float(request.form['distance'])
            terrain = request.form['terrain']
            weather = request.form['weather']

            # Load the feature mapping dictionary to map terrain and weather to model input
            feature_mapping = pd.read_csv('feature_mapping.csv')
            feature_mapping = feature_mapping.set_index('Feature').to_dict()

            # Convert user inputs to model inputs using the feature mapping
            terrain_value = feature_mapping['Value']['Terrain_' + terrain]
            weather_value = feature_mapping['Value']['Weather_' + weather]

            # Prepare input data for prediction
            input_data = [[distance, terrain_value, weather_value]]

            # Make a prediction using the model
            prediction = model.predict(input_data)

            # Return the prediction as JSON
            return jsonify({'prediction': prediction[0]})

        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True , port=2000)
