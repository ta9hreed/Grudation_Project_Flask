from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import nibabel as nib
import tensorflow as tf
import requests
import io
from ml import showPredicts 
import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
import string    
import random
import os


app = Flask(__name__)


# Configuration       
cloudinary.config( 
    cloud_name = "dxbzvepng", 
    api_key = "934452289657873", 
    api_secret = "VDMjFOxUITJpIvW44xzJFQNt2Xw", # Click 'View Credentials' below to copy your API secret
    secure=True
)

def fetch_files_from_urls(urls):
    file_contents = []
    for url in urls:
        print("<<<<<<<"+url+">>>>>>>>>>>")
        url =+1
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad status codes
            print("Starting fetch")
            file_contents.append(response.content)
        except requests.RequestException as e:
            print(f"Error fetching file from {url}: {e}")
            print(file_contents)
    return file_contents


@app.route('/uncompress-and-predict', methods=['POST'])
def uncompress_and_predict():
    try:
        # Get the URLs from the request
        file_urls = request.json.get('file_urls')
        print(file_urls)
        print("this block of URLS")
        if not file_urls or not isinstance(file_urls, list):
            print("No file URLs provided or incorrect format")
            return jsonify({'error': 'No file URLs provided or incorrect format'}), 400
        print("starting")
        result_filename = str(''.join(random.choices(string.ascii_uppercase + string.digits,k=20)))
        showPredicts(file_urls[0], file_urls[1], file_urls[2],result_filename)
        result = []
        for i in range(6):
            # Upload an image
            upload_result = cloudinary.uploader.upload(f'./results/{result_filename}_{i}.png')
            result.append({"secure_url": upload_result["secure_url"], "public_id": upload_result["public_id"]})
            if os.path.exists(f'./results/{result_filename}_{i}.png'):
                os.remove(f'./results/{result_filename}_{i}.png')
        
########
        for i in range(3):
            if os.path.exists(f"./tmp/{file_urls[i]['public_id']}"):
                os.remove(f"./tmp/{file_urls[i]['public_id']}")
        return jsonify({"message":"success","result":result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

if __name__ == '__main__':
    app.run(port=5000, debug=True)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
# from flask import Flask, request, jsonify
# import zlib
# import base64
# import numpy as np
# from tensorflow.keras.models import load_model
# import logging


# app = Flask(__name__)
# logging.basicConfig(level=logging.DEBUG)

# # Load the model
# model = load_model('./models/model/model_per_class.h5')

# def preprocess_data(data):
#     # Convert base64 decoded data to numpy array for prediction
#     # Adjust this preprocessing step as per your data requirements
#     array_data = np.frombuffer(data, dtype=np.float32).reshape((1, -1))
#     return array_data

# @app.route('/uncompress-and-predict', methods=['POST'])
# def uncompress_and_predict():
#     try:
#         files = request.files
#         if not files:
#             return jsonify({'error': 'No files uploaded'}), 400

#         predictions = []
#         for file_key in files:
#             compressed_data = files[file_key].read()
#             try:
#                 uncompressed_data = zlib.decompress(compressed_data)
#             except zlib.error as e:
#                 logging.error(f'Error decompressing file {file_key}: {str(e)}')
#                 return jsonify({'error': f'Error decompressing file {file_key}: {str(e)}'}), 500

#             try:
#                 decoded_data = base64.b64decode(uncompressed_data)
#                 processed_data = preprocess_data(decoded_data)
#                 prediction_result = model.predict(processed_data)
#                 prediction_result_base64 = base64.b64encode(prediction_result.tobytes()).decode()

#                 predictions.append({
#                     'filename': files[file_key].filename,
#                     'prediction': prediction_result.tolist(),
#                     'result': prediction_result_base64
#                 })

#             except Exception as e:
#                 logging.error(f'Error processing file {file_key}: {str(e)}')
#                 return jsonify({'error': f'Error processing file {file_key}: {str(e)}'}), 500

#         return jsonify(predictions), 200

#     except Exception as e:
#         logging.error('Unhandled exception: %s', str(e))
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)



