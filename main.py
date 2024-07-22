import torch
from torchvision import models, transforms
import cv2
import numpy as np

# Load a pre-trained Fast R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Transformation for input frame
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load video input
video_path = 'parking_recording.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'output_video.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a PIL image and apply transformations
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(frame_rgb)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)

    # Draw the bounding boxes on the frame
    for box, score, label in zip(outputs[0]['boxes'], outputs[0]['scores'], outputs[0]['labels']):
        if score > 0.5:  # You can adjust the threshold
            x1, y1, x2, y2 = box.int().numpy()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Write the frame to the output video
    out.write(frame)
    cv2.imwrite('output.jpg',frame)

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
