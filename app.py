from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import cv2
import io

app = Flask(__name__)
CORS(app)

# Load the real trained Random Forest model
model = joblib.load("maize_model.pkl")


# Class names — must match EXACTLY how model was trained in train_model.py
# CATEGORIES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
# So: 0=Blight, 1=Common_Rust, 2=Gray_Leaf_Spot, 3=Healthy
CLASSES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

RECOMMENDATIONS = {
    "Healthy":        "Plant is healthy! Maintain current irrigation and fertilization practices. Monitor regularly.",
    "Blight":         "Apply propiconazole or azoxystrobin fungicide. Remove infected debris. Improve field drainage.",
    "Common_Rust":    "Apply mancozeb or chlorothalonil fungicide. Use resistant varieties next season. Avoid overhead irrigation.",
    "Gray_Leaf_Spot": "Apply strobilurin-based fungicide. Practice crop rotation. Improve plant spacing for air circulation.",
}
def extract_features(img):
    features = []
    img = cv2.resize(img, (100, 100))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [64], [0, 256])
        features.extend(hist.flatten())
    for channel in range(3):
        hist = cv2.calcHist([img], [channel], None, [64], [0, 256])
        features.extend(hist.flatten())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist_gray = cv2.calcHist([gray], [0], None, [64], [0, 256])
    features.extend(hist_gray.flatten())
    for channel in range(3):
        ch = img[:, :, channel].astype(float)
        features.extend([np.mean(ch), np.std(ch), np.min(ch), np.max(ch), np.percentile(ch, 25), np.percentile(ch, 75)])
    edges = cv2.Canny(gray, 100, 200)
    features.append(np.sum(edges > 0) / (100 * 100))
    return np.array(features).reshape(1, -1)

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid image file.")
    return extract_features(img)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        image_bytes = file.read()
        processed = preprocess_image(image_bytes)

        prediction_index = int(model.predict(processed)[0])
        probabilities    = model.predict_proba(processed)[0]

        disease        = CLASSES[prediction_index]
        confidence     = round(float(probabilities[prediction_index]) * 100, 2)
        recommendation = RECOMMENDATIONS.get(disease, "Consult an agronomist.")

        print(f"Predicted: {disease} ({confidence}%) for file: {file.filename}")

        return jsonify({
            "disease":        disease,
            "confidence":     confidence,
            "recommendation": recommendation,
            "healthy":        disease == "Healthy",
            "all_probabilities": {
                CLASSES[i]: round(float(probabilities[i]) * 100, 2)
                for i in range(len(CLASSES))
            }
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running", "model": "Random Forest v1.0", "accuracy": "98.0%"})

if __name__ == "__main__":
    print("Starting AgriScan Flask API...")       # ✅ 4 spaces
    print("Model loaded successfully!")
    print("Classes: Blight=0, Common_Rust=1, Gray_Leaf_Spot=2, Healthy=3")
    app.run(host="0.0.0.0", debug=True, port=5000)
    