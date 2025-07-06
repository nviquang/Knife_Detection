from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(data="./DATA/data.yaml", epochs=10)
