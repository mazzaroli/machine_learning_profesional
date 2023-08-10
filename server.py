# Import necessary libraries
import joblib
import numpy as np

# Import Flask and jsonify for API
from flask import Flask
from flask import jsonify

# Create a Flask application instance
app = Flask(__name__)

# Define an endpoint for predictions (for testing with Postman)
@app.route('/predict', methods=['GET'])
def predict():
    X_test = np.array([7.594444821, 7.479555538, 1.616463184, 1.53352356, 0.796666503, 0.635422587, 0.362012237, 0.315963835, 2.277026653])
    
    # Load the pre-trained machine learning model
    model = joblib.load('./models/best_model_0.932.pkl')
    
    # Make a prediction using the loaded model
    prediction = model.predict(X_test.reshape(1, -1))
    
    # Return the prediction as JSON response
    return jsonify({'prediccion': list(prediction)})

# Run the Flask application if this script is executed directly
if __name__ == "__main__":
    # Load the pre-trained model
    model = joblib.load('./models/best_model_0.932.pkl')
    
    # Start the Flask development server on port 8080
    app.run(port=8080)
