import cv2
import time
from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image
import requests

video_path = 'celticsheat.mp4'
cap = cv2.VideoCapture(video_path)

feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-tiny')
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')

# Define the video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# Process each frame of the video
while cap.isOpened():
    time.sleep(0.1)
    # Read a frame from the video
    ret, frame = cap.read()
    
    # Break the loop if no more frames are available
    if not ret:
        break
    
    # Convert the frame to a PIL Image
    image = Image.fromarray(frame)

    # Perform object detection on the image
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Extract the predicted bounding boxes
    logits = outputs.logits
    bboxes = outputs.pred_boxes

    # Draw the predicted bounding boxes on the frame
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox.tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Write the frame with bounding boxes to the output video
    out.write(frame)

# Release the video capture and writer objects
cap.release()
out.release()
