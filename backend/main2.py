from flask import Flask, request, jsonify
import cv2
import numpy as np
from sklearn.cluster import KMeans
from io import BytesIO
from PIL import Image
import face_alignment
from skimage import img_as_ubyte
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# load face_alignment for face landmarks
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu')

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
    # Resize the face to max dimension for faster processing
    h, w = bgr_image.shape[:2]
    scale = 200 / max(h, w)
    if scale < 1.0:
        bgr_image = cv2.resize(bgr_image, (int(w * scale), int(h * scale)))

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
    if len(pixels) > 1000:
        idx = np.random.choice(len(pixels), 1000, replace=False)
        sample = pixels[idx]
        # if sample is too big we pick a random 1000 pixels
    else:
        sample = pixels
    # cluster pixels into 3 groups, each cluster is a color region
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=3).fit(sample)
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
    
def detect_landmarks(bgr_image):
    # use face_alignment to extract landmarks for face and eyes

    #convert to rgb
    rgb_face = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    rgb_face = img_as_ubyte(rgb_face)  # ensure correct dtype

    preds = fa.get_landmarks(rgb_face) #get landmarks from img
    if preds is None:
        return None
    landmarks = preds[0] #first element in list is landmarks for the face detected
    return {
        "jawline": landmarks[0:17],
        "right_eyebrow": landmarks[17:21],
        "left_eyebrow": landmarks[22:26],
        "left_eye": landmarks[36:42],
        "right_eye": landmarks[42:48],
        "nose": landmarks[27:36],
        "lips": landmarks[48:68]
    }

def classify_face_shape(landmarks):
    # jaw width
    jaw = landmarks["jawline"]
    jaw_left = jaw[0]
    jaw_right = jaw[-1]
    jaw_width = np.linalg.norm(jaw_right - jaw_left)

    # face height
    chin = jaw[8] 
    #approximate forehead height using eyebrow location
    left_eyebrow = min(landmarks["left_eyebrow"][:,1])
    right_eyebrow = min(landmarks["right_eyebrow"][:,1])
    eyebrow_top = min(left_eyebrow, right_eyebrow)

    # forehead will probaly be around 20% of the the rest of the face
    estimated_forehead_y = eyebrow_top - 0.2 * (chin[1] - eyebrow_top)
    face_height = chin[1] - estimated_forehead_y
    jaw_ratio = jaw_width / face_height

    cheekbone_width = np.linalg.norm(jaw[13] - jaw[3])
    cheek_ratio = cheekbone_width / face_height

    if jaw_ratio < 0.8 and cheek_ratio < 0.9:
        return "oval"
    elif jaw_ratio > 1.0:
        return "square"
    elif cheek_ratio > 0.85:
        return "round"
    else:
        return "heart"
    
def classify_eye_shape(landmarks):
    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]

    #ratio of heigtht and width
    def eye_ratio(eye):
        h = np.linalg.norm(eye[1] - eye[5]) # right inner corner - left inner corner
        w = np.linalg.norm(eye[3] - eye[0]) # top inner lid - bottom inner lid
        return h/w
    
    avg_ratio = (eye_ratio(left_eye) + eye_ratio(right_eye))/2
    if avg_ratio < 0.25:
        return "almond"
    elif avg_ratio < 0.35:
        return "hooded"
    else:
        return "round"


    
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

    landmarks = detect_landmarks(face)
    if landmarks is None:
        return jsonify({"error": "no landmarks"})
    
    face_shape = classify_face_shape(landmarks)
    eye_shape = classify_eye_shape(landmarks)

    return jsonify({
        "undertone": undertone,
        "dominant_color_rgb": [int(r), int(g), int(b)],
        "recommended_palette": PALETTES.get(undertone, PALETTES["neutral"]),
        "face_shape" : face_shape,
        "eye_shape": eye_shape
    })
    
if __name__ == '__main__':
    app.run(debug=True)