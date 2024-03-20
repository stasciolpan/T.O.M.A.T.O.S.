import cv2
import os
import subprocess as sp
from ultralytics import YOLO
from picamera2 import Picamera2, Preview
import socket
import math
import pickle
import sys

max_length = 65000
host = sys.argv[1]
port = 6969

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Path to the YOLO model file
model_path = os.path.join('.', 'runs', 'detect', 'train8', 'weights', 'best.pt')
onnx_model_path = os.path.join('.', 'runs', 'detect', 'train8', 'weights', 'best.onnx')

# Create YOLO instance
model = YOLO(model_path)
#model.export(format='onnx') # used for converting to onnx (ONLY USE ONE TIME. COMMNENT AFTER GENERATION)
onnx_model = YOLO(onnx_model_path)

picam2 = Picamera2()
config = picam2.create_preview_configuration({'format': 'RGB888', "size": (640,480)})
picam2.configure(config)
picam2.start()


# Capture video from the webcam
while 1:
    frame = picam2.capture_array()

    # Detect objects using the YOLO model
    results = onnx_model(frame)[0]
    # Iterate through the detection results and extract the bounding boxes and labels
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > 0.50:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{results.names[int(class_id)].upper()} : {score:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    retval, buffer = cv2.imencode(".jpg", frame)

    if retval:
        # convert to byte array
        buffer = buffer.tobytes()
        # get size of the frame
        buffer_size = len(buffer)

        num_of_packs = 1
        if buffer_size > max_length:
            num_of_packs = math.ceil(buffer_size/max_length)

        frame_info = {"packs":num_of_packs}

        # send the number of packs to be expected
        print("Number of packs:", num_of_packs)
        sock.sendto(pickle.dumps(frame_info), (host, port))
        
        left = 0
        right = max_length

        for i in range(num_of_packs):
            print("left:", left)
            print("right:", right)

            # truncate data to send
            data = buffer[left:right]
            left = right
            right += max_length

            # send the frames accordingly
            print(host)
            sock.sendto(data, (host, port))


