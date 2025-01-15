#!/usr/bin/env python3
# pip install opencv-python pandas openalpr

import cv2
import datetime
import pandas as pd
from yolov3_tiny_detector import VehicleDetector
from alpr_module import LicensePlateRecognizer

# Initialize the YOLOv3-tiny model for vehicle detection
vehicle_detector = VehicleDetector("yolov3-tiny.cfg", "yolov3-tiny.weights", "coco.names")

# Initialize the ALPR system
license_plate_recognizer = LicensePlateRecognizer()

# Open the camera feed (CSI camera on Raspberry Pi)
camera_feed = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Use CAP_V4L2 for Raspberry Pi CSI camera

# Output file to store results
output_file = "vehicle_data.csv"

# Initialize DataFrame to store results
data = pd.DataFrame(columns=["date", "time", "num_vehicles", "vehicle_type", "vehicle_color", "vehicle_num"])

while True:
    ret, frame = camera_feed.read()
    if not ret:
        print("Failed to retrieve frame from camera.")
        break

    # Detect vehicles in the frame
    detected_vehicles = vehicle_detector.detect(frame)

    num_vehicles = len(detected_vehicles)
    results = []

    for vehicle in detected_vehicles:
        x, y, w, h, vehicle_type, vehicle_color = vehicle

        # Extract the region of interest (vehicle area)
        vehicle_roi = frame[y:y+h, x:x+w]

        # Recognize the license plate
        license_plate = license_plate_recognizer.recognize(vehicle_roi)

        # Prepare result for this vehicle
        results.append({
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "num_vehicles": num_vehicles,
            "vehicle_type": vehicle_type,
            "vehicle_color": vehicle_color,
            "vehicle_num": license_plate,
        })

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{vehicle_type} | {license_plate}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Append results to the DataFrame
    for result in results:
        data = data.append(result, ignore_index=True)

    # Save the data to the output file
    data.to_csv(output_file, index=False)

    # Display the frame
    cv2.imshow("Traffic Feed", frame)

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera_feed.release()
cv2.destroyAllWindows()
