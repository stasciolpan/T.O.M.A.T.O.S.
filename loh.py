import cv2
import os
import threading
from ultralytics import YOLO
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

# Path to the YOLO model file
model_path = os.path.join('.', 'runs', 'detect', 'train7', 'weights', 'last.pt')

# Create the YOLO object
model = YOLO(model_path)

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Initialize GStreamer
Gst.init(None)

# Create RTSP server
server = GstRtspServer.RTSPServer.new()
mounts = server.get_mount_points()

# Define a custom RTSPMediaFactory subclass
class MyFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, source):
        GstRtspServer.RTSPMediaFactory.__init__(self)
        self.source = source

    def do_create_element(self, url):
        pipeline_str = "( appsrc name=source ! videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency ! rtph264pay name=pay0 pt=96 )"
        pipeline = Gst.parse_launch(pipeline_str)
        self.source = pipeline.get_by_name("source")
        return pipeline

# Create a MyFactory object
factory = MyFactory(None)
mounts.add_factory("/test", factory)

# Create the main loop
mainloop = GLib.MainLoop()

# Attach the server to the default maincontext
server.attach(None)

# Start the loop in a separate thread
thread = threading.Thread(target=mainloop.run)
thread.start()

while (True):
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects using the YOLO model
    results = model(frame)[0]
    # Iterate through the detection results and extract the bounding boxes and labels
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > 0.5:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            label = f"{results.names[int(class_id)].upper()} : {score:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Stream frames over RTSP
    if factory.source is not None:
        factory.source.emit('push-buffer', Gst.Buffer.new_wrapped(frame.tobytes()))

# Release resources and close the window
cap.release()
cv2.destroyAllWindows()