import cv2
import requests
import numpy as np
from flask import make_response

# Load the OpenCV haarcascade_frontalface_default model and load it
resp = requests.get('https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml')
with open('/tmp/haarcascade_frontalface_default.xml', 'wb') as f:
    f.write(resp.content)

face_cascade = cv2.CascadeClassifier('/tmp/haarcascade_frontalface_default.xml')


def find(request):
    # On Options request return specified headers
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, HEAD',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }

        return '', 204, headers

    # Try parsing data as json
    request_json = request.get_json(force=True, silent=True)

    # Check where request data is located and make prediction accordingly
    image_url = None

    if request.args and 'url' in request.args:
        image_url = request.args.get('url')
    elif request_json and 'url' in request_json:
        image_url = request_json['url']
    
    # Default image url if no image provided
    if not image_url:
        image_url = 'https://cdn.vox-cdn.com/thumbor/i06afh0TU9LaYDiLvjxRDwV1am4=/0x0:3049x2048/1820x1213/filters:focal(1333x1562:1819x2048):format(webp)/cdn.vox-cdn.com/uploads/chorus_image/image/63058104/fake_ai_faces.0.png'

    # Load image from url and convert to gray scale
    resp = requests.get(image_url, stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces and draw rectangles
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 4)

    print('Number of faces detected: {}'.format(len(faces)))

    # Create the response
    retval, buffer = cv2.imencode('.png', image)
    response = make_response(buffer.tobytes())
    response.mimetype = 'image/png'
    response.headers['Access-Control-Allow-Origin'] = '*'
    
    return response