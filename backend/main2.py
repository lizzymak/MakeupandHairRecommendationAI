from flask import Flask, request, jsonify
import cv2
import numpy as np
from sklearn.cluster import KMeans
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# used to detect face shape with a little bit of padding for skin tone extraction
def detect_face_region(bgr_image):
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY) #convert image to grayscale
    #returns a list of rectangles where faces are detected
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    #if no faces are detected
    if len(faces) == 0:
        return None
    
    #returns the largest face in the pic by multiplying width and height and sorting in decending order
    x, y, w, h = sorted(faces, key=lambda r: r[2] * r[3], reverse = True)[0]

    # adds padding around the face
    pad_w = int(w * 0.1)
    pad_h = int(h * 0.15)

    #calulcate padded coordinates w/ out going outside image boundary
    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
    x2, y2 = min(bgr_image.shape[1], x + w + pad_w), min(bgr_image.shape[0], y + h + pad_h)

    #return cropped image
    return bgr_image[y1:y2, x1:x2]

def extract_skin_pixels(bgr_image):
    # convert to hsv 
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    #define skin color range
    lower = np.array([0, 15, 60], dtype=np.uint8)
    upper = np.array([50, 170, 255], dtype=np.uint8)
    # hue 0-50, Saturation 15–170, value 60–255 

    #mask for skin pixels if its isnide range its white if its outside then black
    mask = cv2.inRange(hsv, lower, upper)

    # og image will only keep areas that are white in the mask
    skin = cv2.bitwise_and(bgr_image, bgr_image, mask)

    #convert to rgb
    skin_rgb = cv2.cvtColor(skin, cv2.COLOR_BGR2RGB)

    #make image into 1d array rows are each pixel and columns are rgb vals
    pixels = skin_rgb.reshape(-1, 3)

    #filter out all black pixels
    pixels = pixels[np.any(pixels != [0,0,0], axis = 1)]
    return pixels

def dominant_color_kmeans(pixels, n_clusters=3):
    if len(pixels) == 0:
        return None
    if len(pixels) > 5000:
        idx = np.random.choice(len(pixels), 5000, replace=False)
        sample = pixels[idx]
        # if sample is too big we pick a random 5000 pixels
    else:
        sample = pixels
    # cluster pixels into 3 groups, each cluster is a color region
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(sample)
    #pick cluster with most pixels aka the most dominant color
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    largest = labels[np.argmax(counts)]
    #return rgb color for largest color
    dominant = kmeans.cluster_centers_[largest]
    return dominant

def classify_undertone(rgb_color):
    if rgb_color is None:
        return "unknown"
    rgb = np.uint8([[rgb_color]]) # make color a 1x1 img for conversion
    L, a, b = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0][0].astype(float)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0][0].astype(float)
    h = hsv[0]  

    if b - a > 8 or (h < 30 or h > 150):
        return "warm"
    elif a - b > -8 or (60 < h < 150):
        return "cool"
    else:
        return "neutral"
    
PALETTES = {
    "warm": ["#E07A5F", "#F2CC8F", "#81B29A"],
    "cool": ["#6D597A", "#355070", "#E56BC4"],
    "neutral": ["#C9ADA7", "#F2E9E4", "#9A8C98"],
    "unknown": ["#CCCCCC", "#888888"]
}

@app.route("/analyze2", methods=["POST"])
def analyze_image():
    print("FILES:", request.files) 
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    
    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # face detection
    face = detect_face_region(bgr)
    if face is None:
        return jsonify({"error": "No face detected. Try a front-facing photo."}), 400
    
    # skin extraction
    pixels = extract_skin_pixels(face)
    if len(pixels) == 0:
        return jsonify({"error": "No skin-like pixels detected. Try a clearer photo."}), 400
    
    # dominant color
    dominant = dominant_color_kmeans(pixels, n_clusters=3)
    if dominant is None:
        return jsonify({"error": "Could not determine dominant color."}), 400
    
    r, g, b = map(float, dominant)
    undertone = classify_undertone([r, g, b])

    return jsonify({
        "undertone": undertone,
        "dominant_color_rgb": [int(r), int(g), int(b)],
        "recommended_palette": PALETTES.get(undertone, PALETTES["neutral"])
    })
    
if __name__ == '__main__':
    app.run(debug=True)