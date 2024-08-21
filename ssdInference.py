import torch
from torchvision import models, transforms
import cv2
import numpy as np
import time
import psutil

# Load the SSD model architecture
model = models.detection.ssd300_vgg16(weights=None)  # No pretrained weights

# Load weights from the 'model.pth' file
model.load_state_dict(torch.load('model_final.pth'))
model.eval()

# Transformation for input frame
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load video input
video_path = 'parking_recording1.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'ssdOutput.mp4'

# Initialize the VideoWriter with the size of the cropped frames
out = cv2.VideoWriter(output_path, fourcc, 1, (width - 550, height - 200))

parking_positions = {
"a1": [(188, 495), (194, 418), (35, 457), (15, 518)],
    "a2": [(194, 418), (196, 364), (65, 388), (37, 455)],
    "a3": [(197, 365), (203, 320), (84, 337), (65, 387)],
    "a4": [(204, 318), (84, 337), (99, 298), (206, 286)],
    "a5": [(149, 177), (210, 175), (212, 165), (152, 168)],
    "a6": [(215, 154), (155, 157), (152, 168), (212, 165)],
    "a7": [(215, 154), (155, 157), (162, 147), (215, 144)],
    "a8": [(162, 147), (215, 144), (216, 138), (163, 140)],
    "a9": [(216, 138), (163, 140), (168, 130), (216, 127)],
    "b1": [(340, 340), (439, 319), (418, 288), (326, 305)],
    "b2": [(418, 288), (326, 305), (316, 275), (401, 260)],
    "b3": [(316, 275), (401, 260), (388, 238), (310, 251)],
    "b4": [(388, 238), (310, 251), (304, 228), (375, 217)],
    "b5": [(304, 228), (375, 217), (365, 200), (299, 211)],
    "b6": [(365, 200), (299, 211), (294, 195), (357, 186)],
    "b7": [(294, 195), (357, 186), (348, 173), (290, 178)],
    "r1": [(205, 284), (103, 298), (113, 272), (206, 262)],
    "r2": [(210, 240), (124, 247), (130, 229), (210, 222)],
    "r3": [(130, 229), (210, 222), (210, 205), (138, 211)],
    "r4": [(213, 192), (143, 196), (145, 185), (214, 181)]}

tot_parking = len(parking_positions.keys())

while cap.isOpened():
    start_time = time.time()  # Start timing

    ret, frame = cap.read()
    if not ret:
        break

    rows, cols, _ = frame.shape

    # Crop frame
    frame = frame[200: rows, 300: cols - 250]

    # Convert the frame to a PIL image and apply transformations
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(frame_rgb)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)

    # Store car center points
    cars_center_pos = []

    # Draw the bounding boxes on the frame
    for box, score, label in zip(outputs[0]['boxes'], outputs[0]['scores'], outputs[0]['labels']):
        if score > 0.12 and label in [3]:  # Labels 3 correspond to 'car'
            x1, y1, x2, y2 = box.int().numpy()
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cars_center_pos.append((cx, cy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
            cv2.putText(frame, 'car', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Store names of occupied parkings
    occupied_parkings = set()

    # Draw parking positions
    for parking_name, parking_pos in parking_positions.items():
        centroid = np.array(parking_pos).mean(axis=0)
        cx = int(centroid[0])
        cy = int(centroid[1])

        # Drawing parking spots
        polygon_color = (25, 155, 180)
        if "r" in parking_name:
            polygon_color = (180, 20, 15)

        cv2.putText(frame, parking_name, (cx - 10, cy + 5), cv2.FONT_HERSHEY_PLAIN, 0.7, polygon_color, 1)
        cv2.polylines(frame, np.array([parking_pos]), True, polygon_color, 2)

        # Detect if there is a car inside the parking
        for car_center in cars_center_pos:
            cx, cy = car_center
            is_inside = cv2.pointPolygonTest(np.array(parking_pos),
                                             (int(cx), int(cy)), False)
            if is_inside > 0:
                print("{} is occupied".format(parking_name))
                occupied_parkings.add(parking_name)
                break

    print("Occupied parkings: ", occupied_parkings)
    available_parkings = tot_parking - len(occupied_parkings)

    # Display available parking info
    cv2.rectangle(frame, (0, 0), (500, 35), (255, 255, 255), -1)
    cv2.putText(frame, "Available parkings {}/{}".format(available_parkings, tot_parking), (15, 30),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    # Write the frame to the output video
    out.write(frame)

    # Show frame with bounding boxes and parking info
    cv2.imshow("Parking Detection", frame)

    end_time = time.time()  # End timing
    frame_time = end_time - start_time  # Calculate frame processing time
    cpu_usage = psutil.cpu_percent(interval=1)  # Get CPU usage over the interval

    print(f"Time per frame: {frame_time:.4f} seconds")
    print(f"CPU Usage: {cpu_usage}%")

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break

cap.release()
out.release()
cv2.destroyAllWindows()
