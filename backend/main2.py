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
    
    