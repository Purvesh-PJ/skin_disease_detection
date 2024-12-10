from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

def setup_prediction_routes(app: Flask):

    @app.route('/predict', methods=['POST'])
    def prediction():
        # Print request headers for debugging
        print("Request Headers:", request.headers)

        # Check if the request contains an image file
        if 'image' not in request.files:
            print("No image part in the request.")
            return jsonify({"error": "No image file provided"}), 400

        # Get the image file from the request
        image_file = request.files['image']
        
        # Log the file name and content type
        print("File Name:", image_file.filename)
        print("Content Type:", image_file.content_type)

        # Check if the file has a valid name
        if image_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Read the actual file data (as bytes)
        file_data = image_file.read()  # Read the file data into memory
        print("File Data (first 100 bytes):", file_data[:100])

        # Optional: Save the file to a specific directory
        temp_dir = 'uploads'  # Update this path as needed
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # Secure the filename to prevent directory traversal attacks
        filename = secure_filename(image_file.filename)
        file_path = os.path.join(temp_dir, filename)

        # Save the file
        with open(file_path, 'wb') as f:
            f.write(file_data)  # Save the file data

        return jsonify({"message": "Image received and processed successfully", "filename": filename})

# Make sure to call the setup function to register routes
setup_prediction_routes(app)

if __name__ == '__main__':
    app.run(debug=True)



        


       

