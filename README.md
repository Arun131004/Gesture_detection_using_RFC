# Gesture Recognition Web Application

This web application allows you to train and test hand gesture recognition models directly in your browser. Built with Flask, OpenCV, MediaPipe, and scikit-learn.

## Features

- View all available trained gestures
- Add new gestures using your webcam
- Remove existing gestures
- Test the model in real-time using your webcam

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask application:
```bash
python app.py
```

3. Open your browser and navigate to `http://localhost:5000`

## Usage

### Adding a New Gesture

1. Enter the name of the gesture in the input field
2. Click "Start Recording"
3. Perform the gesture in front of your webcam
4. Click "Stop Recording" when done
5. The gesture will be added to the model and the page will refresh

### Testing Gestures

1. Click "Start Testing" in the Test Model section
2. Perform gestures in front of your webcam
3. The predicted gesture will be displayed in real-time
4. Click "Stop Testing" when done

### Removing Gestures

1. Click the X button next to any gesture in the Available Gestures section
2. Confirm the deletion when prompted

## Technical Details

- Hand tracking is done using MediaPipe Hands
- The model uses a Random Forest Classifier for gesture recognition
- Images are processed in real-time using OpenCV
- The web interface is built with Flask and styled using Tailwind CSS 