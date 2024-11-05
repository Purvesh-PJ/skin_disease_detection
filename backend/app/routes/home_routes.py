from flask import Flask

def setup_home_routes(app: Flask):
    @app.route('/')

    def home():
        return "Welcome to the Skin Disease Detection API!"