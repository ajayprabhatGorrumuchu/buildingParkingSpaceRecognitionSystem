import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time
import psutil

# Load the YOLOv8x model
model = YOLO('yolov8x.pt')

# Get class names and create the dictionary for name to ID mapping
class_names = model.names
name_to_id = {name: id for id, name in class_names.items()}

# Get the IDs for 'car' and 'truck'
car_class_id = name_to_id.get('car')
truck_class_id = name_to_id.get('truck')

if car_class_id is None or truck_class_id is None:
    raise ValueError("Class names 'car' or 'truck' not found in the model.")

# Load video input
video_path = 'parking_recording1.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'yolov8Output.mp4'

# Initialize the VideoWriter with the size of the cropped frames
out = cv2.VideoWriter(output_path, fourcc, fps, (width - 550, height - 200))

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

    # Convert the frame to PIL image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform inference
    results = model(img)

    # Store car center points
    cars_center_pos = []

    # Process each result
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

        # Draw results on the frame
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            conf = confidences[i]
            cls = int(class_ids[i])

            # Filter for car and truck
            if cls == car_class_id or cls == truck_class_id:
                label = f"{class_names[cls]} {conf:.2f}"

                # Convert coordinates to integer
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw the bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Calculate center position
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cars_center_pos.append((cx, cy))
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

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
            is_inside = cv2.pointPolygonTest(np.array(parking_pos), (int(cx), int(cy)), False)
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
    cv2.imshow("YOLOv8 Parking Detection", frame)

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
