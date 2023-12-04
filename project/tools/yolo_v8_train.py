from ultralytics import YOLO

# load a pretrained model
model = YOLO('yolov8n.pt')  

# Train the model
results = model.train(data='./yolo_v8.yaml', epochs=1, imgsz=640)
