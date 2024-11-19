# Droidcam-Drowsiness-Detection-Setup


## Files to Include

### `README.md`

#### Content for README.md

```markdown
# DroidCam Drowsiness Detection System

This project is an updated implementation of a drowsiness detection system that uses DroidCam as a webcam feed source for detecting drowsiness. The aim is to establish communication between a PC server and a mobile device using the same IP to share real-time video data for drowsiness analysis.

## Overview

The system uses a video stream provided by DroidCam to capture real-time video, processes the frames using `dlib`'s 68-point facial landmark predictor to detect eyes, and calculates the eye aspect ratio (EAR). The EAR value is then used to determine if the user is drowsy by comparing it against a predefined threshold.

**Technologies Used:**
- Python
- OpenCV
- Dlib
- DroidCam

## How It Works
- **DroidCam Integration**: The system connects to the DroidCam feed via an IP address to acquire real-time video.
- **Dlib Facial Landmarks**: The project uses `dlib`'s 68-point facial landmark predictor to identify and track the eyes.
- **EAR Calculation**: The Eye Aspect Ratio (EAR) is calculated to determine whether the eyes are closed for a prolonged period, indicating drowsiness.
- **Real-Time Alerts**: Alerts are displayed directly on the video feed if drowsiness is detected.

## Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed. The following libraries are required:

- `dlib` - for facial landmark detection
- `opencv-python` (`cv2`) - for video stream handling
- `scipy` - for spatial distance calculation
- `numpy` - for numerical operations

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd droidcam-drowsiness-detection
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the `shape_predictor_68_face_landmarks.dat` file from [dlib's Model Zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the root of the project folder.

### Usage

To start the detection system, run the following command:

```bash
python droidcam_drowsiness_detection.py
```

Press `q` to quit the webcam feed.

### Note
- Make sure DroidCam is running and that your PC and mobile device are on the same network.
- Ensure the correct IP address is used in the code.

## Future Work

This project serves as a test implementation for real-time video streaming using DroidCam and will be expanded further to add additional communication methods between devices.
```
