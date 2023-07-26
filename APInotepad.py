from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('my_model.pkl')

# Load the DataFrame for locations
locations = pd.read_csv('locations.csv')

@app.route('/predict', methods=['POST'])
def predict_price():
    try:
        data = request.json
        print(data)

        # Retrieve the input values
        location = data['location']
        sqft = float(data['sqft'])
        bath = float(data['bath'])
        bhk = float(data['bhk'])

        # Get the location index from the locations DataFrame
        loc_index = locations[locations['location'] == location].index[0]

        # Create the input array for prediction
        x = np.zeros(len(locations))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk
        if loc_index >= 0:
            x[loc_index] = 1

        # Make the prediction using your model
        price = model.predict([x])[0]

        # Return the predicted price as a JSON response
        return jsonify({
            'success': True,
            'price': price
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
