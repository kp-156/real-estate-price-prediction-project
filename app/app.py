import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = joblib.load('../model/price-prediction-model.pkl')

# Load the DataFrame for locations - encoded as one hot columns
column_df = pd.read_csv('../data/columns.csv')


@app.route('/predict', methods=['POST'])
def predict_price():
    try:
        data = request.json
        # Retrieve the input values

        location = data['location']
        sqft = float(data['sqft'])
        bath = float(data['bath'])
        bhk = float(data['bhk'])

        # Get the location index from the locations DataFrame
        loc_index = column_df.index[column_df['columns'] == location].tolist()[0]
        # Create the input array for prediction
        x = np.zeros(len(column_df['columns']))
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
        print(str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/', methods=['GET'])
def hello():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
