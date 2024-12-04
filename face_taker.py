import numpy as np
import json
import cv2
import os

def create_directory(directory: str) -> None:
    """
    Create a directory if it doesn't exist.

    Parameters:
        directory (str): The path of the directory to be created.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_face_id(directory: str) -> int:
    """
    Get the first available face identifier.

    Parameters:
        directory (str): The path of the directory of images.
    """
    user_ids = []
    for filename in os.listdir(directory):
        try:
            
            number = int(os.path.split(filename)[-1].split("-")[1])
            user_ids.append(number)
        except ValueError:
            continue

    user_ids = sorted(list(set(user_ids)))
    max_user_id = 1 if len(user_ids) == 0 else max(user_ids) + 1
    return max_user_id

def save_name(face_id: int, face_name: str, filename: str) -> None:
    """
    Save name and face id to JSON.

    Parameters:
        face_id (int): The identifier of the user.
        face_name (str): The name of the user.
        filename (str): The filename where to save the data.
    """
    names_json = {}
    if os.path.exists(filename):
        with open(filename, 'r') as fs:
            names_json = json.load(fs)
    
    names_json[face_id] = face_name
    with open(filename, 'w') as fs:
        json.dump(names_json, fs, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    directory = 'images'
    cascade_classifier_filename = 'haarcascade_frontalface_default.xml'
    names_json_filename = 'names.json'

    
    create_directory(directory)
    
    
    faceCascade = cv2.CascadeClassifier(cascade_classifier_filename)
    
    
    cam = cv2.VideoCapture(0)
    
    
    cam.set(3, 640)
    cam.set(4, 480)
    
    
    count = 0
    face_name = input('\nEnter user name and press <return> -->  ')
    face_id = get_face_id(directory)
    save_name(face_id, face_name, names_json_filename)
    print('\n[INFO] Initializing face capture. Look at the camera and wait...')

    while True:
        
        ret, img = cam.read()
    
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
        
        for (x, y, w, h) in faces:
            
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
        
            count += 1

        
            cv2.imwrite(f'./images/Users-{face_id}-{count}.jpg', gray[y:y+h, x:x+w])
    
            
            cv2.imshow('image', img)
    
        
        k = cv2.waitKey(100) & 0xff
        if k == 27:  
            break
    
        
        elif count >= 30:
            break
    
    print('\n[INFO] Success! Exiting Program.')
    
    
    cam.release()
    
    
    cv2.destroyAllWindows()
