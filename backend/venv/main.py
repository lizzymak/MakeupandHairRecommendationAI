from PIL import Image
from flask import Flask, request, jsonify
import cv2
import numpy as np
import io

app = Flask(__name__) #app instance

def extract_dominant_color(image, k=3):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert image from bgr to rgb
    image = image.reshape((-1, 3)) #transforms image array into an array with 3 columns(rgb)
    image = np.float32(image) # change data type of array

    # using k means to find dominant color clusters
    criteria = (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2) #stop if 20 iterations or error of 0.2 is reached
    _,  labels, centers = cv2.kmeans(image, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # find the largest cluster (most dominant color)
    _, counts = np.unique(labels, return_counts=True) #extracts each unique element inteh array and how many times it appears
    dominant = centers[np.argmax(counts)]
    return tuple(map(int, dominant))

# classify undertone based on dominant color
def determine_undertone(rgb):
    r, g, b = rgb
    if b > r and b > g:
        return "cool"
    elif r > b and g > b:
        return "warm"
    else:
        return "neutral"
    
@app.route('/analyze', methods=['POST'])
def analyze_image():
    # Get uploaded file
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400

    # Read image into OpenCV
    image_bytes = file.read()
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Resize for faster processing
    img = cv2.resize(img, (400, 400))

    # Get dominant color
    dominant = extract_dominant_color(img)

    # Determine undertone
    undertone = determine_undertone(dominant)

    # Recommend palette (simplified example)
    recommendations = {
        "cool": ["rose pink", "mauve", "platinum blonde"],
        "warm": ["peach", "coral", "golden blonde"],
        "neutral": ["nude", "taupe", "chocolate brown"]
    }

    return jsonify({
        "dominant_color": dominant,
        "undertone": undertone,
        "recommended_palette": recommendations[undertone]
    })

if __name__ == '__main__':
    app.run(debug=True)

