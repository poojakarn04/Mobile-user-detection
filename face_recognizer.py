import cv2
import numpy as np
import json
import os
import time  # Import the time module for the timer
import requests  # Import requests for ClickSend API
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Specify the class ID for cell phones (COCO dataset: class ID 67)
cell_phone_class_id = [67]  # Only detect cell phones

# Load face recognizer and training data
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

# Load the face cascade
face_cascade_Path = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(face_cascade_Path)

# Font for displaying text
font = cv2.FONT_HERSHEY_SIMPLEX

# Load the names
id = 0
names = ['None']
with open('names.json', 'r') as fs:
    names = json.load(fs)
    names = list(names.values())

# Initialize webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

# Set minimum width and height for face detection
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# Timer variables
timer_started = False
start_time = None

# Replace with your ClickSend credentials
USERNAME = '22ise007@bnmit.in'  # Your ClickSend username
API_KEY = 'A97C0646-8482-8108-19A6-4E36C40C488B'  # Your ClickSend API Key

# Function to send SMS using ClickSend
def send_sms(to_number, message):
    # ClickSend API endpoint
    url = "https://rest.clicksend.com/v3/sms/send"
    
    # Payload for the SMS
    payload = {
        "messages": [
            {
                "source": "python",
                "body": message,
                "to": to_number
            }
        ]
    }

    # Basic Auth for ClickSend
    auth = (USERNAME, API_KEY)

    # Send request
    try:
        response = requests.post(url, json=payload, auth=auth)
        response_data = response.json()  # Parse the JSON response
        if response.status_code == 200:
            print("Message sent successfully:", response_data)
            return "Message sent successfully"
        else:
            print("Failed to send SMS:", response_data)
            return f"Failed to send SMS: {response_data}"
    except Exception as e:
        print("Error occurred:", str(e))
        return f"Error occurred: {str(e)}"

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Run the YOLO model for cell phone detection
    results = model(img, conf=0.4, classes=cell_phone_class_id)
    cell_phone_detected = False  # Flag to track cell phone detection

    # Variable to store the name of the person using a cell phone
    person_using_phone = None

    # Draw bounding boxes for YOLO detections (cell phones)
    for result in results[0].boxes:
        # Accessing the detection data from each result
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
        confidence = result.conf[0]  # Confidence score (Tensor)

        # Convert Tensor to float and format as percentage
        confidence_percent = round(confidence.item() * 100)  # Use .item() to get the float value

        if confidence_percent > 55:  # Confidence threshold for cell phone detection
            cell_phone_detected = True  # Set flag to true if the cell phone is detected with high confidence
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "Cell Phone", (x1, y1 - 10), font, 0.9, (0, 255, 0), 2)
            cv2.putText(img, f"{confidence_percent}%", (x1, y2 + 20), font, 0.9, (0, 255, 0), 2)

    # Start the timer if a cell phone is detected
    if cell_phone_detected:
        if not timer_started:
            timer_started = True
            start_time = time.time()  # Record the start time
    else:
        timer_started = False  # Reset timer if no cell phone is detected
        start_time = None

    # Only perform face detection if a cell phone is detected
    if cell_phone_detected:
        # Detect faces in the frame
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        # Draw bounding boxes for faces and recognize them
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            if confidence > 51:
                try:
                    name = names[id]
                    person_using_phone = name  # Store the detected person's name
                    confidence_face = f"{round(confidence)}%"
                except IndexError:
                    name = "Unknown"
                    confidence_face = "N/A"
            else:
                name = "Unknown"
                confidence_face = "N/A"

            cv2.putText(img, name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, confidence_face, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    # Display the message and send SMS after 5 seconds
    if timer_started and start_time:
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        if elapsed_time > 5:  # Check if the timer exceeds 5 seconds
            if person_using_phone:
                message = f"Warning!, {person_using_phone} , you have been using mobile from 5 minutes!"
                print(message)  # Print the message in the terminal
                cv2.putText(img, message, (10, 50), font, 1, (0, 0, 255), 2)  # Overlay on the video feed
                
                # Send SMS after 5 seconds
                phone_number = "+918123856196"  # Replace with actual phone number
                send_sms(phone_number, message)  # Send the message via ClickSend

    # Display the frame with face and cell phone detections
    cv2.imshow('camera', img)

    # Exit if the ESC key is pressed
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

print("\n [INFO] Exiting Program.")
cam.release()
cv2.destroyAllWindows()
