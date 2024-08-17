from flask import Flask, request, jsonify
import cv2 as cv
import numpy as np
import io
import firebase_admin
from firebase_admin import credentials, storage, db
import secrets
import string
import datetime
import urllib.parse  # For URL encoding

app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = credentials.Certificate('project_files\pothole-99ab7-firebase-adminsdk-dyony-0830dced16.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'pothole-99ab7.appspot.com',
    'databaseURL': 'https://pothole-99ab7-default-rtdb.firebaseio.com/'
})

# Initialize Firebase Storage and Realtime Database
bucket = storage.bucket()
ref = db.reference('data')

# Load YOLO model
net1 = cv.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
model1 = cv.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

# Define the threshold parameters
Conf_threshold = 0.5
NMS_threshold = 0.4

def generate_random_id(length=20):
    characters = string.ascii_lowercase + string.digits
    return ''.join(secrets.choice(characters) for _ in range(length))

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files or 'location' not in request.form:
        return jsonify({"error": "Missing image or location data"}), 400
    
    # Retrieve image and location data from request
    image_file = request.files['image']
    location = request.form['location']
    
    # Read image from file
    npimg = np.frombuffer(image_file.read(), np.uint8)
    frame = cv.imdecode(npimg, cv.IMREAD_COLOR)

    if frame is not None:
        # Pothole detection
        classes, scores, boxes = model1.detect(frame, Conf_threshold, NMS_threshold)
        for (classid, score, box) in zip(classes, scores, boxes):
            label = "pothole"
            x, y, w, h = box
            recarea = w * h
            area = frame.shape[1] * frame.shape[0]

            if len(scores) != 0 and scores[0] >= 0.7:
                if (recarea / area) <= 0.1 and box[1] < 600:
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv.putText(frame, "%" + str(round(scores[0] * 100, 2)) + " " + label, (box[0], box[1] - 10), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

                    # Resize the image to reduce file size
                    new_width = 640  # Example width, adjust as needed
                    aspect_ratio = frame.shape[1] / frame.shape[0]
                    new_height = int(new_width / aspect_ratio)
                    resized_frame = cv.resize(frame, (new_width, new_height))

                    # Compress the image
                    _, img_encoded = cv.imencode('.jpg', resized_frame, [cv.IMWRITE_JPEG_QUALITY, 70])  # Adjust quality as needed
                    img_bytes = io.BytesIO(img_encoded.tobytes())
                    
                    # Generate a unique ID
                    random_id = generate_random_id()

                    # Upload image to Firebase Storage
                    latlng = location.replace(',', '_')  # Format lat-long for filename
                    filename = f'pothole_{latlng}_{random_id}.jpg'
                    blob = bucket.blob(filename)
                    blob.upload_from_file(img_bytes, content_type='image/jpeg')

                    # Construct the URL in the desired format
                    encoded_filename = urllib.parse.quote(filename)  # URL-encode the filename
                    url = f'https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{encoded_filename}?alt=media'

                    # Save image URL and location to Firebase Realtime Database
                    ref.child(random_id).set({
                        'lat': location.split(',')[0],
                        'long': location.split(',')[1],
                        'url': url
                    })
                    
                    # Print success message
                    print(f"Image successfully uploaded to Firebase Storage: {url}")
                    print(f"Data saved to Firebase Realtime Database: {{'lat': {location.split(',')[0]}, 'long': {location.split(',')[1]}, 'url': {url}}}")

    return '', 204  # No content response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
