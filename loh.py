import cv2
import os
import subprocess as sp
from ultralytics import YOLO
from picamera2 import Picamera2, Preview

# Path to the YOLO model file
model_path = os.path.join('.', 'runs', 'detect', 'train7', 'weights', 'best.pt')
onnx_model_path = os.path.join('.', 'runs', 'detect', 'train7', 'weights', 'best.onnx')

# Create camera context
picam2 = Picamera2()
config = picam2.create_preview_configuration({'format': 'RGB888', "size": (640,480)})
picam2.configure(config)
picam2.start()

# Create the YOLO object
model = YOLO(model_path)
#model.export(format='onnx')
onnx_model = YOLO(onnx_model_path)

# Capture video from the webcam
proc = None
while 1:
    frame = picam2.capture_array()
    # Downsample the frame
    #frame = cv2.resize(frame, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR);
    
    # Define the FFmpeg command
    command = [
        'ffmpeg',
        '-y',
        '-re',
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', '{}x{}'.format(*frame.shape[1::-1]),
        #'-r', str(cap.get(cv2.CAP_PROP_FPS)),
        '-r', '5',
        '-i', '-',
        '-an',
        '-vcodec', 'libx264',
        '-b:v', '5000k',
        '-f', 'flv',
        'rtmp://localhost/live/tomato'
    ]

    # Create the FFmpeg process
    if proc is None:
        proc = sp.Popen(command, stdin=sp.PIPE)

    # Detect objects using the YOLO model
    results = onnx_model(frame)[0]
    # Iterate through the detection results and extract the bounding boxes and labels
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > 0.60:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{results.names[int(class_id)].upper()} : {score:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    # Write the frame to the FFmpeg process
    try:
        proc.stdin.write(frame.tobytes())
    except BrokenPipeError:
        print("FFmpeg process has terminated. Exiting.")
        break

