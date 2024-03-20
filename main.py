from ultralytics import YOLO

model = YOLO("/home/heiwa/git/tomat/T.O.M.A.T.O.S./runs/detect/train8/weights/last.pt")
results = model.train(data="config_tomatoless.yaml", epochs=10, device=0)
