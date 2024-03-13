import cv2
import os
import subprocess as sp
from ultralytics import YOLO
from picamera2 import Picamera2, Preview

# Path to the YOLO model file
model_path = os.path.join('.', 'runs', 'detect', 'train7', 'weights', 'last.pt')

# Create camera context
picam2 = Picamera2()
#picam2.configure(picam2.create_preview_configuration({"size": (640, 480)}))
#picam2.start_preview(Preview.QTGL)
picam2.start()

# Create the YOLO object
model = YOLO(model_path)

# Capture video from the webcam
proc = None
while True:
    frame = picam2.capture_array()
    frame = frame[:,:,:3]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Downsample the frame
    #frame = cv2.resize(frame, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR);
    
    # Define the FFmpeg command
    command = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', '{}x{}'.format(*frame.shape[1::-1]),
        #'-r', str(cap.get(cv2.CAP_PROP_FPS)),
        '-r', '15',
        '-i', '-',
        '-an',
        '-vcodec', 'mpeg4',
        '-b:v', '5000k',
        '-f', 'rtsp',
        'rtsp://localhost:8554/tomato'
    ]

    # Create the FFmpeg process
    if proc is None:
        proc = sp.Popen(command, stdin=sp.PIPE)

    # Detect objects using the YOLO model
    results = model(frame)[0]
    # Iterate through the detection results and extract the bounding boxes and labels
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > 0.65:
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

# Release the webcam
#cap.release()
#cv2.destroyAllWindows()
