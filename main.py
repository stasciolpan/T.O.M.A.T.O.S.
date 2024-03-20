from ultralytics import YOLO

model = YOLO("/home/heiwa/git/tomat/T.O.M.A.T.O.S./runs/detect/train7/weights/last.pt")
results = model.train(data="config.yaml", epochs=300, device=0, resume=True)
