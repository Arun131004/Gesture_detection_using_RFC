import os
import cv2
import pickle
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import base64
import time

app = Flask(__name__)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Constants
DATA_DIR = 'data'
MODEL_PATH = 'model.p'
DATA_PATH = 'data.pickle'
DATASET_SIZE = 100  # Number of images per gesture

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def get_available_gestures():
    ensure_data_dir()
    return [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

def process_hand_landmarks(frame):
    data_aux, x_, y_ = [], [], []
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x_.append(hand_landmarks.landmark[i].x)
                y_.append(hand_landmarks.landmark[i].y)
            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                data_aux.append(hand_landmarks.landmark[i].y - min(y_))
        return data_aux, True
    return None, False

def create_dataset():
    """Creates dataset from all images in the data directory"""
    data = []
    labels = []
    
    for dir_ in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_)
        if not os.path.isdir(dir_path):
            continue
            
        print(f"Processing gesture: {dir_}")
        for img_path in os.listdir(dir_path):
            data_aux = []
            x_ = []
            y_ = []
            
            img = cv2.imread(os.path.join(dir_path, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x_.append(hand_landmarks.landmark[i].x)
                        y_.append(hand_landmarks.landmark[i].y)
                    for i in range(len(hand_landmarks.landmark)):
                        data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                        data_aux.append(hand_landmarks.landmark[i].y - min(y_))
                data.append(data_aux)
                labels.append(dir_)
    
    return np.array(data), np.array(labels)

def train_model():
    """Creates dataset and trains a new model with all available data"""
    print("Starting model training...")
    start_time = time.time()
    
    # Create dataset
    data, labels = create_dataset()
    
    if len(data) == 0 or len(labels) == 0:
        print("No valid training data found")
        return False
        
    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    
    # Evaluate model
    y_predict = model.predict(x_test)
    accuracy = accuracy_score(y_predict, y_test)
    
    print(f"Model trained with accuracy: {accuracy * 100:.2f}%")
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    print(f"Total samples: {len(data)}, Classes: {len(set(labels))}")
    
    # Save model and dataset
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({
            'model': model,
            'accuracy': accuracy,
            'classes': list(set(labels)),
            'training_size': len(data)
        }, f)
    
    with open(DATA_PATH, 'wb') as f:
        pickle.dump({
            'data': data,
            'labels': labels
        }, f)
    
    return True

@app.route('/')
def index():
    gestures = get_available_gestures()
    model_info = None
    
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model_dict = pickle.load(f)
            model_info = {
                'accuracy': f"{model_dict['accuracy'] * 100:.2f}%",
                'classes': model_dict['classes'],
                'training_size': model_dict['training_size']
            }
    
    return render_template('index.html', gestures=gestures, model_info=model_info)

@app.route('/api/gestures')
def get_gestures():
    gestures = get_available_gestures()
    return jsonify(gestures)

@app.route('/api/add_gesture', methods=['POST'])
def add_gesture():
    data = request.json
    gesture_name = data['name']
    frames = data['frames']  # Base64 encoded frames
    
    gesture_dir = os.path.join(DATA_DIR, gesture_name)
    if not os.path.exists(gesture_dir):
        os.makedirs(gesture_dir)
    
    # Save frames as images
    saved_frames = 0
    for idx, frame_data in enumerate(frames):
        if idx >= DATASET_SIZE:  # Limit to dataset size
            break
            
        try:
            # Convert base64 to image
            image_data = base64.b64decode(frame_data.split(',')[1])
            with open(os.path.join(gesture_dir, f'{idx}.jpg'), 'wb') as f:
                f.write(image_data)
            saved_frames += 1
        except Exception as e:
            print(f"Error saving frame {idx}: {str(e)}")
    
    print(f"Saved {saved_frames} frames for gesture {gesture_name}")
    
    # Retrain model
    success = train_model()
    return jsonify({
        "success": success,
        "frames_saved": saved_frames
    })

@app.route('/api/remove_gesture/<gesture>', methods=['DELETE'])
def remove_gesture(gesture):
    gesture_dir = os.path.join(DATA_DIR, gesture)
    if os.path.exists(gesture_dir):
        import shutil
        shutil.rmtree(gesture_dir)
        print(f"Removed gesture directory: {gesture}")
        
        # Retrain model if there are still gestures
        if len(get_available_gestures()) > 0:
            success = train_model()
        else:
            # Remove model files if no gestures left
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            if os.path.exists(DATA_PATH):
                os.remove(DATA_PATH)
            success = True
            
        return jsonify({"success": success})
    return jsonify({"success": False, "error": "Gesture not found"})

@app.route('/api/predict', methods=['POST'])
def predict():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not found"})
    
    try:
        # Load model
        with open(MODEL_PATH, 'rb') as f:
            model_dict = pickle.load(f)
            model = model_dict['model']
        
        # Get frame data
        data = request.json
        frame_data = data['frame']  # Base64 encoded frame
        
        # Convert base64 to image
        image_data = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process frame
        data_aux, has_hand = process_hand_landmarks(frame)
        if has_hand:
            prediction = model.predict([np.asarray(data_aux)])
            confidence = model.predict_proba([np.asarray(data_aux)]).max()
            return jsonify({
                "prediction": prediction[0],
                "confidence": f"{confidence * 100:.2f}%"
            })
        return jsonify({"prediction": "No hand detected"})
        
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    ensure_data_dir()
    app.run(debug=True) 