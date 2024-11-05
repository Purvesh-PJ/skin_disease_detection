from flask import Flask, request, jsonify
from flask_cors import CORS
from app.routes.home_routes import setup_home_routes
from app.routes.prediction_routes import setup_prediction_routes

app = Flask(__name__) # instance flask app.
CORS(app) # Enable CORS for all routes.

# Register routes
setup_home_routes(app) # home route
setup_prediction_routes(app) # predection route


if __name__ == '__main__':
    app.run(debug=True)
