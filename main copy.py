import cv2
import numpy as np
import onnxruntime as ort

ort_session = ort.InferenceSession('model.onnx')
input_shape = (320, 320)

def post_process(outputs, frame):
    boxes = outputs[0]

    if boxes is None or len(boxes) == 0:
        print("No objects to display.")
        return frame

    if boxes.ndim == 3 and boxes.shape[2] >= 6:
        for box in boxes[0]:
            xcenter, ycenter, width, height, conf, cls = box[:6]
            xcenter, ycenter, width, height = int(xcenter.item()), int(ycenter.item()), int(width.item()), int(height.item())
            conf = float(conf.item())
            cls = int(cls.item())
            x1 = xcenter - width
            x2 = xcenter + width
            y1 = ycenter - height
            y2 = ycenter + height
            
            if conf > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    else:
        print("Invalid model output:", boxes.shape if boxes is not None else "None")

    return frame

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not retrieve frame.")
        break
    
    if frame is None:
        print("Error: Could not load image.")
        exit()

    resized_frame = cv2.resize(frame, input_shape)
    input_image = cv2.dnn.blobFromImage(resized_frame, 1/255.0, input_shape, swapRB=True, crop=False)
    outputs = ort_session.run(None, {"input": input_image})
    resized_frame = post_process(outputs, resized_frame)

    cv2.imshow('YOLOv7 Detection', resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
