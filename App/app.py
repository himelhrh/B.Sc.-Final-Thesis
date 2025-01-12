from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
MODEL_PATH = '/Users/hasiburrahman/Desktop/Deploy/covid19_vgg19_model.h5'
model = load_model(MODEL_PATH)
class_labels = ['Covid', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']  # Replace with your classes

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)



# Preprocess image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Match model input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array  # No scaling as per your scalar(img)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle file upload
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)  # Save the uploaded file

            # Preprocess and predict
            img_array = preprocess_image(filepath)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=-1)
            confidence = np.max(predictions)

            # Get class name
            predicted_class_name = class_labels[predicted_class[0]]

            return render_template("index.html", 
                                   prediction=predicted_class_name, 
                                   confidence=f"{confidence * 100:.2f}%", 
                                   image_url=filepath)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)