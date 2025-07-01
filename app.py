import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Konfigurasi ---
model_path = 'coffee_bean_classifier_model.h5'
IMG_HEIGHT = 150
IMG_WIDTH = 150
class_labels = {0: 'biji bagus', 1: 'biji rusak'}

# --- Model Initialization ---
loaded_model = None
try:
    loaded_model = load_model(model_path)
    print(f"Model '{model_path}' loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model from '{model_path}': {e}")
    print("Ensure the model file exists and is not corrupted.")
    exit()

# --- Flask Application Initialization ---
app = Flask(__name__)
CORS(app)

# --- API Routes ---

@app.route('/')
def home():
    return "Coffee Bean Quality Classification API is Running! Use the /predict endpoint for predictions."

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print("[DEBUG] No 'file' part in request.")
        return jsonify({'error': 'No file part in the request. Ensure you send a file with the key "file".'}), 400

    file = request.files['file']

    if file.filename == '':
        print("[DEBUG] No file selected (empty filename).")
        return jsonify({'error': 'No file selected.'}), 400

    if file:
        try:
            print(f"[DEBUG] Received file: {file.filename}, Content-Type: {file.content_type}")

            # Membaca gambar dari request
            img_bytes = file.read()
            print(f"[DEBUG] Image bytes length: {len(img_bytes)}")

            # Coba deteksi apakah ini file gambar yang valid sebelum membuka
            # Beberapa browser mungkin mengirimkan Content-Type yang salah
            if not file.content_type.startswith('image/'):
                print(f"[DEBUG] Warning: Content-Type is not image: {file.content_type}")
                # Anda bisa menambahkan validasi di sini untuk menolak non-gambar
                # return jsonify({'error': 'Uploaded file is not an image.'}), 400

            # --- Bagian yang sering menyebabkan Invalid Argument ---
            # Pastikan Pillow dapat membaca bytes ini
            try:
                img = Image.open(io.BytesIO(img_bytes))
                print("[DEBUG] Image opened successfully with Pillow.")
            except Exception as img_open_e:
                print(f"[ERROR] Failed to open image with Pillow: {img_open_e}")
                return jsonify({'error': f'Failed to open image: {str(img_open_e)}. Is it a valid image file?'}), 400

            # Konversi ke RGB (penting jika gambar asli Grayscale atau RGBA)
            img = img.convert('RGB')
            print("[DEBUG] Image converted to RGB.")

            # Preprocessing gambar: resize dan normalisasi
            img = img.resize((IMG_WIDTH, IMG_HEIGHT))
            print(f"[DEBUG] Image resized to {IMG_WIDTH}x{IMG_HEIGHT}.")

            img_array = np.array(img) / 255.0
            print("[DEBUG] Image converted to numpy array and normalized.")

            img_array = np.expand_dims(img_array, axis=0)
            print("[DEBUG] Batch dimension added.")

            # Melakukan prediksi
            prediction_raw = loaded_model.predict(img_array)[0][0]
            print(f"[DEBUG] Prediction raw output: {prediction_raw}")

            predicted_class_index = int(prediction_raw > 0.5)
            predicted_label = class_labels.get(predicted_class_index, "Unknown")
            confidence = float(prediction_raw) if predicted_class_index == 1 else float(1 - prediction_raw)

            print(f"[DEBUG] Predicted label: {predicted_label}, Confidence: {confidence:.4f}")

            return jsonify({
                'prediction': predicted_label,
                'confidence': f"{confidence:.4f}",
                'raw_output': float(prediction_raw)
            })

        except Exception as e:
            # Ini akan menangkap error "Invalid argument" dan lainnya
            print(f"[ERROR] Error processing request within try block: {e}")
            return jsonify({'error': f'Failed to process image or perform prediction: {str(e)}'}), 500
    else:
        print("[DEBUG] File object is None (should not happen if file.filename is not empty).")
        return jsonify({'error': 'Invalid or empty file.'}), 400

# --- Run the Flask Application ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)