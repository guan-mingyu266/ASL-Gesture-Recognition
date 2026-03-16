from ultralytics import YOLO

yolo = YOLO("runs/detect/train/weights/best.pt", task="detect")
result = yolo(source=0, show=True)
