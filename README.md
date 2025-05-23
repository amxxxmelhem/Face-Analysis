
# Face Analysis

**Face Analysis** is a powerful desktop GUI application for performing face-based analysis using deep learning and computer vision techniques. It supports real-time webcam input and image uploads to detect and analyze facial attributes including **emotion**, **age**, **gender**, **skin tone**, and **beauty metrics** like symmetry, proportions, and evenness.

## Features

- 📸 **Image Input**: Upload images or capture using webcam.
- 😊 **Emotion Detection**: Detects primary emotion with confidence score.
- 🧑 **Demographics Detection**: Predicts age group and gender.
- 💅 **Beauty Analysis**: Scores based on facial proportions, symmetry, and skin quality.
- 🎨 **Skin Tone Detection**: Estimates Fitzpatrick scale skin tone.
- 🧾 **Summary Report**: Generates a detailed facial analysis report.
- 🧠 **Deep Learning**: Utilizes pre-trained CNNs for emotion, age, and gender detection.
- 🎯 **Confidence Threshold**: Adjustable confidence filter for predictions.

## Installation

### Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

### Clone the Repository

```bash
git clone https://github.com/yourusername/face-analysis.git
cd face-analysis
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Additional Downloads Required

Download the following model files and place them in the project directory:

- `emotion_model.h5`
- `opencv_face_detector.pbtxt`
- `opencv_face_detector_uint8.pb`
- `age_deploy.prototxt`
- `age_net.caffemodel`
- `gender_deploy.prototxt`
- `gender_net.caffemodel`

You can find these from public model zoos or refer to the documentation in the repo.

## Usage

Run the application with:

```bash
python FaceAnalysis.py
```

### Instructions

1. Click **"📁 Upload Image"** or **"📸 Open Camera"** to select an input.
2. Press **"⏺ Capture"** if using the camera.
3. Hit **"🔍 Analyze Face"** to run detection and analysis.
4. Navigate tabs for:
   - **Summary**
   - **Emotion & Demographics**
   - **Beauty Analysis**
   - **Full Report**

## Tech Stack

- **GUI**: Tkinter
- **Image Processing**: OpenCV, Pillow
- **ML Framework**: TensorFlow/Keras
- **Face Landmarking**: MediaPipe
- **DL Models**: Custom-trained CNN + OpenCV DNNs

## File Structure

```
FaceAnalysis.py           # Main application file
emotion_model.h5          # Emotion detection model
age_net.caffemodel        # Age prediction model
gender_net.caffemodel     # Gender prediction model
opencv_face_detector.pb*  # Face detection models
```

## Acknowledgements

- [MediaPipe by Google](https://github.com/google/mediapipe)
- [OpenCV](https://opencv.org/)
- Pre-trained models from public repositories.

## License

MIT License. See [LICENSE](LICENSE) for more details.
