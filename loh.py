import cv2
import os
from ultralytics import YOLO

# Path-ul către fișierul cu modelul YOLO v8qq
model_path = os.path.join('.', 'runs', 'detect', 'train7', 'weights', 'last.pt')

# Crearea obiectului YOLO
model = YOLO(model_path)

# Capturarea video de la camera
cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()
    if not ret:
        break

    # Detectarea obiectelor folosind modelul YOLO v8
    #results = model(frame)
    results = model(frame)[0]
    # Iterarea prin rezultatele de detectare și extragerea chenarelor și etichetelor
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > 0.5:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Afișare frame-uri cu rezultatele detectării
    cv2.imshow('YOLO v8 Detection', frame)

    # Întreruperea ciclului dacă tasta 'q' este apăsată
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Eliberarea resurselor și închiderea ferestrei
cap.release()
cv2.destroyAllWindows()