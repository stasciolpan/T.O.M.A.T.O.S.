import cv2
import socket
import pickle
import numpy as np

host = '0.0.0.0'
port = 6969
max_length = 65000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((host, port))
sock.settimeout(30)
frame_info = None
buffer = None
frame = None

print("-> waiting for connection")

while True:
    try:
        data, address = sock.recvfrom(max_length)
    except socket.timeout:
        print("ERROR: Socket timeout achieved")
        break
    if len(data) < 100:
        frame_info = pickle.loads(data)

        if frame_info:
            nums_of_packs = frame_info["packs"]

            for i in range(nums_of_packs):
                data, address = sock.recvfrom(max_length)

                if i == 0:
                    buffer = data
                else:
                    buffer += data

            frame = np.frombuffer(buffer, dtype=np.uint8)
            frame = frame.reshape(frame.shape[0], 1)

            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            frame = cv2.flip(frame, 1)
            
            if frame is not None and type(frame) == np.ndarray:
                print("Connection exists and frame recieved")
                cv2.imshow("Stream", frame)
                if cv2.waitKey(1) == 27:
                    break
                
print("goodbye")
