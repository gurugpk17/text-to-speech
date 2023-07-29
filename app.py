# app.py
from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.h5')

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = np.array(data['input']).reshape(1, -1)

    # Make predictions using the model
    prediction = model.predict(input_data)

    target_names = ['setosa', 'versicolor', 'virginica']
    result = target_names[prediction[0]]

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run()
