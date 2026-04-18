import cv2
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

DATADIR = "C:/Users/91824/Desktop/Maize Images/data"
CATEGORIES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

def extract_features(img):
    features = []
    img = cv2.resize(img, (100, 100))

    # 1. HSV Color Histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [64], [0, 256])
        features.extend(hist.flatten())

    # 2. BGR Color Histogram
    for channel in range(3):
        hist = cv2.calcHist([img], [channel], None, [64], [0, 256])
        features.extend(hist.flatten())

    # 3. Grayscale Histogram
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist_gray = cv2.calcHist([gray], [0], None, [64], [0, 256])
    features.extend(hist_gray.flatten())

    # 4. Statistical features per channel
    for channel in range(3):
        ch = img[:, :, channel].astype(float)
        features.extend([
            np.mean(ch),
            np.std(ch),
            np.min(ch),
            np.max(ch),
            np.percentile(ch, 25),
            np.percentile(ch, 75),
        ])

    # 5. Edge density
    edges = cv2.Canny(gray, 100, 200)
    features.append(np.sum(edges > 0) / (100 * 100))

    return np.array(features)

training_data = []
labels = []

print("Loading and extracting features...")
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    class_num = CATEGORIES.index(category)
    count = 0
    for img_name in os.listdir(path):
        try:
            img = cv2.imread(os.path.join(path, img_name))
            if img is None:
                continue

            # Original image
            training_data.append(extract_features(img))
            labels.append(class_num)

            # Flipped horizontally
            training_data.append(extract_features(cv2.flip(img, 1)))
            labels.append(class_num)

            count += 1
        except:
            pass
    print(f"{category}: {count} images loaded ({count*2} with augmentation)")

print(f"\nTotal samples: {len(training_data)}")
print(f"Features per image: {len(training_data[0])}")
print("Training model — please wait...")

model = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    min_samples_split=4,
    n_jobs=-1,
    random_state=42,
    oob_score=True,
    verbose=1
)

model.fit(training_data, labels)
print(f"\nOOB Accuracy: {round(model.oob_score_ * 100, 2)}%")
joblib.dump(model, "maize_model.pkl")
print("Model saved successfully!")