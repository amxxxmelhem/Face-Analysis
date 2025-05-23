import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk, ImageOps
import cv2
import numpy as np
import math
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import mediapipe as mp
from threading import Thread

class FaceAnalysisApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Analysis Pro")
        self.master.geometry("1200x900")
        self.master.configure(bg="#f5f5f5")
        
        # Initialize models
        self.load_models()
        
        # Initialize MediaPipe for beauty analysis
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            max_num_faces=10,  # Increased for multi-face support
            refine_landmarks=True, 
            min_detection_confidence=0.7
        )
        
        # Beauty metrics weights
        self.BEAUTY_WEIGHTS = {
            'face_ratio': 0.4,
            'symmetry': 0.3,
            'skin_evenness': 0.2,
            'feature_alignment': 0.1
        }
        
        # Constants
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Confidence threshold
        self.confidence_threshold = 0.4
        
        # Camera variables
        self.camera_on = False
        self.camera = None
        self.current_camera_frame = None
        
        # Create GUI
        self.create_widgets()
    
    def load_models(self):
        try:
            # Emotion model
            EMOTION_MODEL_PATH = 'emotion_model.h5'
            if not os.path.exists(EMOTION_MODEL_PATH):
                raise FileNotFoundError(f"Emotion model file '{EMOTION_MODEL_PATH}' not found!")
            
            # Age and gender models
            FACE_PROTO = "opencv_face_detector.pbtxt"
            FACE_MODEL = "opencv_face_detector_uint8.pb"
            AGE_PROTO = "age_deploy.prototxt"
            AGE_MODEL = "age_net.caffemodel"
            GENDER_PROTO = "gender_deploy.prototxt"
            GENDER_MODEL = "gender_net.caffemodel"
            
            self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
            
            self.emotion_model = load_model(EMOTION_MODEL_PATH)
            self.face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
            self.age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
            self.gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading models: {str(e)}")
            self.master.destroy()
    
    def create_widgets(self):
        # Header Frame
        header_frame = tk.Frame(self.master, bg="#6a0dad")
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.header = tk.Label(
            header_frame, 
            text="Face Analysis Pro", 
            font=("Arial", 24, "bold"), 
            fg="white",
            bg="#6a0dad",
            pady=15
        )
        self.header.pack()
        
        # Main Container Frame
        main_frame = tk.Frame(self.master, bg="#f5f5f5")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left Panel (Image Upload/Camera)
        left_panel = tk.Frame(main_frame, bg="#f5f5f5")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Image Canvas
        self.canvas = tk.Canvas(
            left_panel, 
            width=400, 
            height=400, 
            bg="white",
            highlightthickness=2,
            highlightbackground="#6a0dad"
        )
        self.canvas.pack(pady=(0, 20))
        
        # Camera Frame
        camera_frame = tk.Frame(left_panel, bg="#f5f5f5")
        camera_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Camera Toggle Button
        self.camera_btn = tk.Button(
            camera_frame,
            text="📸 Open Camera",
            command=self.toggle_camera,
            bg="#2196F3",
            fg="white",
            font=("Arial", 10, "bold"),
            relief=tk.FLAT
        )
        self.camera_btn.pack(side=tk.LEFT, padx=5)
        
        # Capture Button (hidden initially)
        self.capture_btn = tk.Button(
            camera_frame,
            text="⏺ Capture",
            command=self.capture_image,
            bg="#FF5722",
            fg="white",
            font=("Arial", 10, "bold"),
            state=tk.DISABLED,
            relief=tk.FLAT
        )
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        # Upload Button
        self.upload_btn = tk.Button(
            left_panel, 
            text="📁 Upload Image", 
            command=self.upload_image,
            bg="#6a0dad",
            fg="white",
            font=("Arial", 10, "bold"),
            relief=tk.FLAT
        )
        self.upload_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Analyze Button (hidden initially)
        self.analyze_btn = tk.Button(
            left_panel,
            text="🔍 Analyze Face",
            command=self.analyze_image,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            state=tk.DISABLED,
            relief=tk.FLAT
        )
        self.analyze_btn.pack(fill=tk.X, pady=(0, 20))
        
        # Confidence Threshold Slider
        threshold_frame = tk.Frame(left_panel, bg="#f5f5f5")
        threshold_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(
            threshold_frame,
            text="Confidence Threshold:",
            font=("Arial", 10),
            bg="#f5f5f5"
        ).pack(side=tk.LEFT)
        
        self.threshold_slider = tk.Scale(
            threshold_frame,
            from_=0.1,
            to=0.9,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            command=self.update_threshold,
            bg="#f5f5f5"
        )
        self.threshold_slider.set(self.confidence_threshold)
        self.threshold_slider.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Right Panel (Results)
        right_panel = tk.Frame(main_frame, bg="#f5f5f5")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # Results Notebook (Tabbed interface)
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Summary Tab
        self.summary_tab = tk.Frame(self.notebook, bg="#f5f5f5")
        self.notebook.add(self.summary_tab, text="Summary")
        
        # Emotion & Demographics Tab
        self.emotion_tab = tk.Frame(self.notebook, bg="#f5f5f5")
        self.notebook.add(self.emotion_tab, text="Emotion & Demographics")
        
        # Beauty Analysis Tab
        self.beauty_tab = tk.Frame(self.notebook, bg="#f5f5f5")
        self.notebook.add(self.beauty_tab, text="Beauty Analysis")
        
        # Full Report Tab
        self.report_tab = tk.Frame(self.notebook, bg="#f5f5f5")
        self.notebook.add(self.report_tab, text="Full Report")
        
        # Initialize results UI components
        self.init_summary_tab()
        self.init_emotion_tab()
        self.init_beauty_tab()
        self.init_report_tab()
    
    def init_summary_tab(self):
        # Score Frame
        score_frame = tk.Frame(self.summary_tab, bg="#f5f5f5")
        score_frame.pack(fill=tk.X, pady=20)
        
        self.score_header = tk.Label(
            score_frame,
            text="Your Analysis Summary",
            font=("Arial", 16, "bold"),
            bg="#f5f5f5"
        )
        self.score_header.pack()
        
        # Emotion Summary
        emotion_frame = tk.Frame(self.summary_tab, bg="#f5f5f5")
        emotion_frame.pack(fill=tk.X, pady=10)
        
        self.emotion_summary = tk.Label(
            emotion_frame,
            text="Primary Emotion: ",
            font=("Arial", 12),
            bg="#f5f5f5"
        )
        self.emotion_summary.pack(anchor="w")
        
        # Demographic Summary
        demo_frame = tk.Frame(self.summary_tab, bg="#f5f5f5")
        demo_frame.pack(fill=tk.X, pady=10)
        
        self.demo_summary = tk.Label(
            demo_frame,
            text="Demographics: ",
            font=("Arial", 12),
            bg="#f5f5f5"
        )
        self.demo_summary.pack(anchor="w")
        
        # Beauty Score
        beauty_frame = tk.Frame(self.summary_tab, bg="#f5f5f5")
        beauty_frame.pack(fill=tk.X, pady=10)
        
        self.beauty_score_label = tk.Label(
            beauty_frame,
            text="Beauty Score: ",
            font=("Arial", 12),
            bg="#f5f5f5"
        )
        self.beauty_score_label.pack(anchor="w")
        
        # Quick Assessment
        assessment_frame = tk.Frame(self.summary_tab, bg="#f5f5f5")
        assessment_frame.pack(fill=tk.X, pady=20)
        
        self.assessment_header = tk.Label(
            assessment_frame,
            text="Quick Assessment",
            font=("Arial", 16, "bold"),
            bg="#f5f5f5"
        )
        self.assessment_header.pack()
        
        self.assessment_text = tk.Text(
            assessment_frame,
            wrap=tk.WORD,
            font=("Arial", 11),
            bg="white",
            height=5,
            padx=10,
            pady=10
        )
        self.assessment_text.pack(fill=tk.X)
    
    def init_emotion_tab(self):
        # Emotion Results Frame
        emotion_frame = tk.Frame(self.emotion_tab, bg="#f5f5f5")
        emotion_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Faces Detected
        self.faces_label = tk.Label(
            emotion_frame,
            text="Faces Detected: ",
            font=("Arial", 12),
            bg="#f5f5f5"
        )
        self.faces_label.pack(anchor="w")
        
        # Results Treeview
        self.results_tree = ttk.Treeview(
            emotion_frame,
            columns=("Emotion", "Confidence", "Gender", "Age"),
            show="headings",
            height=5
        )
        self.results_tree.heading("Emotion", text="Emotion")
        self.results_tree.heading("Confidence", text="Confidence")
        self.results_tree.heading("Gender", text="Gender")
        self.results_tree.heading("Age", text="Age")
        self.results_tree.column("Emotion", width=100)
        self.results_tree.column("Confidence", width=100)
        self.results_tree.column("Gender", width=100)
        self.results_tree.column("Age", width=100)
        self.results_tree.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Processed Image Display
        self.processed_canvas = tk.Canvas(
            emotion_frame,
            width=400,
            height=300,
            bg="white",
            highlightthickness=1,
            highlightbackground="#cccccc"
        )
        self.processed_canvas.pack(pady=10)
    
    def init_beauty_tab(self):
        # Beauty Results Frame
        beauty_frame = tk.Frame(self.beauty_tab, bg="#f5f5f5")
        beauty_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Faces Detected
        self.beauty_faces_label = tk.Label(
            beauty_frame,
            text="Beauty Analysis Results:",
            font=("Arial", 12),
            bg="#f5f5f5"
        )
        self.beauty_faces_label.pack(anchor="w")
        
        # Beauty Results Treeview
        self.beauty_tree = ttk.Treeview(
            beauty_frame,
            columns=("Face", "Score", "Symmetry", "Skin Tone"),
            show="headings",
            height=5
        )
        self.beauty_tree.heading("Face", text="Face #")
        self.beauty_tree.heading("Score", text="Beauty Score")
        self.beauty_tree.heading("Symmetry", text="Symmetry")
        self.beauty_tree.heading("Skin Tone", text="Skin Tone")
        self.beauty_tree.column("Face", width=80)
        self.beauty_tree.column("Score", width=100)
        self.beauty_tree.column("Symmetry", width=100)
        self.beauty_tree.column("Skin Tone", width=120)
        self.beauty_tree.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Score Visualization Frame
        self.beauty_score_canvas = tk.Canvas(
            beauty_frame, 
            width=300, 
            height=80, 
            bg="white",
            highlightthickness=0
        )
        self.beauty_score_canvas.pack(pady=10)
        
        # Skin Tone Frame
        skin_frame = tk.Frame(beauty_frame, bg="#f5f5f5")
        skin_frame.pack(fill=tk.X, pady=10)
        
        self.skin_header = tk.Label(
            skin_frame,
            text="Skin Analysis",
            font=("Arial", 12, "bold"),
            bg="#f5f5f5"
        )
        self.skin_header.pack(anchor="w")
        
        skin_subframe = tk.Frame(skin_frame, bg="#f5f5f5")
        skin_subframe.pack()
        
        self.skin_label = tk.Label(
            skin_subframe,
            text="Primary Skin Tone: ",
            font=("Arial", 11),
            bg="#f5f5f5"
        )
        self.skin_label.pack(side=tk.LEFT)
        
        self.skin_color_display = tk.Canvas(
            skin_subframe, 
            width=50, 
            height=50, 
            bg="white",
            highlightthickness=1
        )
        self.skin_color_display.pack(side=tk.LEFT, padx=10)
        
        # Facial Proportions Frame
        proportions_frame = tk.LabelFrame(
            beauty_frame,
            text="Average Facial Proportions",
            font=("Arial", 12, "bold"),
            bg="#f5f5f5"
        )
        proportions_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.face_ratio_label = tk.Label(
            proportions_frame,
            text="Face Ratio (H/W): ",
            font=("Arial", 11),
            bg="#f5f5f5"
        )
        self.face_ratio_label.pack(anchor="w")
        
        self.symmetry_label = tk.Label(
            proportions_frame,
            text="Symmetry Score: ",
            font=("Arial", 11),
            bg="#f5f5f5"
        )
        self.symmetry_label.pack(anchor="w")
        
        self.alignment_label = tk.Label(
            proportions_frame,
            text="Feature Alignment: ",
            font=("Arial", 11),
            bg="#f5f5f5"
        )
        self.alignment_label.pack(anchor="w")
    
    def init_report_tab(self):
        # Full Report Text
        self.report_text = tk.Text(
            self.report_tab,
            wrap=tk.WORD,
            font=("Arial", 11),
            bg="white",
            padx=10,
            pady=10
        )
        self.report_text.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = tk.Scrollbar(self.report_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.report_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.report_text.yview)
    
    def toggle_camera(self):
        if not self.camera_on:
            # Open camera
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            
            self.camera_on = True
            self.camera_btn.config(text="📷 Close Camera")
            self.capture_btn.config(state=tk.NORMAL)
            self.upload_btn.config(state=tk.DISABLED)
            self.analyze_btn.config(state=tk.DISABLED)
            
            # Start camera preview thread
            self.camera_thread = Thread(target=self.update_camera_preview, daemon=True)
            self.camera_thread.start()
        else:
            # Close camera
            self.camera_on = False
            self.camera_btn.config(text="📸 Open Camera")
            self.capture_btn.config(state=tk.DISABLED)
            self.upload_btn.config(state=tk.NORMAL)
            
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            
            # Clear camera preview
            self.canvas.delete("all")
            self.canvas.create_rectangle(0, 0, 400, 400, fill="white", outline="")
    
    def update_camera_preview(self):
        while self.camera_on and self.camera is not None:
            ret, frame = self.camera.read()
            if ret:
                # Convert to RGB and resize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (400, 400))
                
                # Store current frame for capture
                self.current_camera_frame = frame
                
                # Convert to PhotoImage and update canvas
                img = Image.fromarray(frame)
                self.camera_photo = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor="nw", image=self.camera_photo)
            
            # Small delay to prevent high CPU usage
            self.master.update()
            self.master.after(30)
    
    def capture_image(self):
        if self.current_camera_frame is not None:
            # Convert frame to PIL Image
            img = Image.fromarray(self.current_camera_frame)
            
            # Save to temporary file
            self.image_path = "captured_image.jpg"
            img.save(self.image_path)
            
            # Display the captured image
            self.display_image(self.image_path)
            
            # Enable analysis button
            self.analyze_btn.config(state=tk.NORMAL)
            
            messagebox.showinfo("Success", "Image captured successfully!")
    
    def update_threshold(self, value):
        self.confidence_threshold = float(value)
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.analyze_btn.config(state=tk.NORMAL)
            messagebox.showinfo("Ready", "Image uploaded successfully! Click 'Analyze Face' to proceed.")
    
    def display_image(self, path):
        self.image = Image.open(path)
        self.image = ImageOps.fit(self.image, (400, 400), method=Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
    
    def analyze_image(self):
        if not hasattr(self, 'image_path'):
            messagebox.showerror("Error", "Please upload or capture an image first")
            return
            
        # Process the image for emotion, age, gender detection
        image = Image.open(self.image_path)
        result_image, emotion_results, num_faces = self.detect_emotions_and_demographics(image)
        
        # Process for beauty analysis
        frame = cv2.imread(self.image_path)
        beauty_results = []
        avg_beauty_score = 0
        primary_skin_tone = None
        primary_skin_color = None
        avg_features = {
            'face_ratio': 0,
            'symmetry': 0,
            'skin_evenness': 0,
            'feature_alignment': 0
        }
        
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                # Process all faces found
                for i, face_landmarks in enumerate(results.multi_face_landmarks):
                    landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) 
                               for lm in face_landmarks.landmark]
                    
                    # Skin analysis
                    skin_roi = self.get_skin_roi(frame, landmarks)
                    skin_tone, skin_color = self.analyze_skin_tone(skin_roi)
                    
                    # Beauty analysis
                    beauty_features = self.extract_beauty_features(landmarks, skin_roi)
                    beauty_score = self.predict_beauty_score(beauty_features)
                    
                    # Store results for this face
                    beauty_results.append({
                        'face_num': i+1,
                        'score': beauty_score,
                        'skin_tone': skin_tone,
                        'skin_color': skin_color,
                        'features': beauty_features
                    })
                    
                    # Update averages
                    avg_beauty_score += beauty_score
                    for key in avg_features:
                        avg_features[key] += beauty_features[key]
                    
                    # Set primary skin tone (first face)
                    if i == 0:
                        primary_skin_tone = skin_tone
                        primary_skin_color = skin_color
                
                # Calculate averages
                if beauty_results:
                    avg_beauty_score /= len(beauty_results)
                    for key in avg_features:
                        avg_features[key] /= len(beauty_results)
        
        # Update all UI components with results
        self.update_summary_tab(emotion_results, avg_beauty_score, primary_skin_tone, num_faces)
        self.update_emotion_tab(result_image, emotion_results, num_faces)
        self.update_beauty_tab(beauty_results, avg_beauty_score, primary_skin_tone, primary_skin_color, avg_features)
        self.update_report_tab(emotion_results, beauty_results, avg_beauty_score, primary_skin_tone, avg_features, num_faces)
    
    def preprocess_face(self, face, target_size=(48, 48)):
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, target_size)
        face_array = img_to_array(gray)
        face_array = np.expand_dims(face_array, axis=0)
        face_array /= 255.0
        return face_array
    
    def predict_emotion(self, face):
        processed_face = self.preprocess_face(face)
        preds = self.emotion_model.predict(processed_face, verbose=0)[0]
        return self.emotion_labels[np.argmax(preds)], np.max(preds)
    
    def predict_age_gender(self, face):
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = self.gender_list[gender_preds[0].argmax()]
        
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = self.age_list[age_preds[0].argmax()]
        
        return gender, age
    
    def highlight_face(self, net, frame, conf_threshold=0.7):
        frame_opencv_dnn = frame.copy()
        frame_height = frame_opencv_dnn.shape[0]
        frame_width = frame_opencv_dnn.shape[1]
        blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()
        face_boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                face_boxes.append([x1, y1, x2, y2])
        return face_boxes
    
    def detect_faces(self, frame):
        # Try DNN face detector first (more accurate)
        face_boxes = self.highlight_face(self.face_net, frame)
        
        # Fall back to Haar cascade if DNN doesn't detect faces
        if not face_boxes:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            face_boxes = [[x, y, x+w, y+h] for (x, y, w, h) in faces]
        
        return face_boxes
    
    def detect_emotions_and_demographics(self, image, confidence_threshold=None):
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
            
        frame = np.array(image)
        
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        face_boxes = self.detect_faces(frame)
        results = []
        output_frame = frame.copy()
        padding = 20
        
        for face_box in face_boxes:
            x1, y1, x2, y2 = face_box
            x1, y1 = max(0, x1-padding), max(0, y1-padding)
            x2, y2 = min(frame.shape[1]-1, x2+padding), min(frame.shape[0]-1, y2+padding)
            
            face_roi = frame[y1:y2, x1:x2]
            
            try:
                # Predict emotion
                emotion, confidence = self.predict_emotion(face_roi)
                if confidence < confidence_threshold:
                    continue
                    
                # Predict age and gender
                gender, age = self.predict_age_gender(face_roi)
                
                # Draw results
                color = (0, 255, 0)
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                
                # Display emotion with confidence
                cv2.putText(
                    output_frame, 
                    f"{emotion} ({confidence:.2f})", 
                    (x1, y1-50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    color, 
                    2
                )
                
                # Display age and gender
                cv2.putText(
                    output_frame, 
                    f"{gender}, {age}", 
                    (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 255), 
                    2
                )
                
                results.append({
                    "emotion": emotion,
                    "confidence": float(confidence),
                    "gender": gender,
                    "age": age,
                    "position": (int(x1), int(y1), int(x2-x1), int(y2-y1))
                })
            except Exception as e:
                continue
        
        return cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB), results, len(face_boxes)
    
    def update_summary_tab(self, emotion_results, avg_beauty_score, skin_tone, num_faces):
        # Clear previous results
        self.assessment_text.delete(1.0, tk.END)
        
        # Update emotion summary
        primary_emotion = max(set([r['emotion'] for r in emotion_results]), key=[r['emotion'] for r in emotion_results].count) if emotion_results else 'N/A'
        self.emotion_summary.config(text=f"Primary Emotion: {primary_emotion}")
        
        # Update demographic summary
        if emotion_results:
            genders = [r['gender'] for r in emotion_results]
            ages = [r['age'] for r in emotion_results]
            demo_text = f"Demographics: {', '.join(genders)} | Age Groups: {', '.join(ages)}"
            self.demo_summary.config(text=demo_text)
        else:
            self.demo_summary.config(text="Demographics: No faces detected")
        
        # Update beauty score if available
        if avg_beauty_score > 0:
            self.beauty_score_label.config(text=f"Average Beauty Score: {avg_beauty_score:.2f}/1.0")
            self.draw_score_meter(avg_beauty_score)
        
        # Update assessment
        assessment = self.generate_assessment(emotion_results, avg_beauty_score, num_faces)
        self.assessment_text.insert(tk.END, assessment)
    
    def update_emotion_tab(self, result_image, results, num_faces):
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Update faces detected label
        self.faces_label.config(text=f"Faces Detected: {num_faces}")
        
        # Add results to treeview
        for res in results:
            self.results_tree.insert("", "end", values=(
                res['emotion'],
                f"{res['confidence']:.2f}",
                res['gender'],
                res['age']
            ))
        
        # Display processed image
        self.display_processed_image(result_image)
    
    def update_beauty_tab(self, beauty_results, avg_beauty_score, skin_tone, skin_color, avg_features):
        # Clear previous results
        for item in self.beauty_tree.get_children():
            self.beauty_tree.delete(item)
        
        # Add beauty results to treeview
        for res in beauty_results:
            self.beauty_tree.insert("", "end", values=(
                res['face_num'],
                f"{res['score']:.2f}",
                f"{res['features']['symmetry']:.2f}",
                res['skin_tone']
            ))
        
        # Update beauty score visualization
        self.beauty_score_canvas.delete("all")
        if avg_beauty_score > 0:
            self.draw_score_meter(avg_beauty_score)
        
        # Update skin tone display
        if skin_tone:
            self.skin_label.config(text=f"Primary Skin Tone: {skin_tone}")
            self.skin_color_display.delete("all")
            self.skin_color_display.create_rectangle(
                0, 0, 50, 50, 
                fill=skin_color,
                outline=""
            )
        
        # Update facial proportions
        if avg_features['face_ratio'] > 0:
            self.face_ratio_label.config(
                text=f"Face Ratio (H/W): {avg_features['face_ratio']:.2f} (Ideal ≈ 1.62)"
            )
            self.symmetry_label.config(
                text=f"Symmetry Score: {avg_features['symmetry']:.2f}/1.0"
            )
            self.alignment_label.config(
                text=f"Feature Alignment: {avg_features['feature_alignment']:.2f}/1.0"
            )
    
    def update_report_tab(self, emotion_results, beauty_results, avg_beauty_score, skin_tone, avg_features, num_faces):
        self.report_text.delete(1.0, tk.END)
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        primary_emotion = max(set([r['emotion'] for r in emotion_results]), key=[r['emotion'] for r in emotion_results].count) if emotion_results else 'N/A'
        avg_confidence = sum(r['confidence'] for r in emotion_results)/len(emotion_results) if emotion_results else 0
        
        report = f"""
        {' FACE ANALYSIS REPORT '.center(50, '=')}
        
        Date: {current_time}
        
        [SUMMARY]
        Faces Detected: {num_faces}
        Primary Emotion: {primary_emotion}
        Average Confidence: {avg_confidence:.1%}
        """
        
        if avg_beauty_score > 0:
            report += f"Average Beauty Score: {avg_beauty_score:.2f}/1.0\n"
            report += f"Primary Skin Tone: {skin_tone}\n"
        
        report += """
        [EMOTION & DEMOGRAPHICS]
        """
        
        for i, res in enumerate(emotion_results, 1):
            report += f"""
            Face {i}:
            • Emotion: {res['emotion']} ({res['confidence']:.1%})
            • Gender: {res['gender']}
            • Age Group: {res['age']}
            """
        
        if beauty_results:
            report += """
            [BEAUTY ANALYSIS]
            """
            
            for res in beauty_results:
                report += f"""
                Face {res['face_num']}:
                • Beauty Score: {res['score']:.2f}/1.0
                • Symmetry: {res['features']['symmetry']:.2f}/1.0
                • Skin Tone: {res['skin_tone']}
                """
            
            report += """
            [AVERAGE FACIAL PROPORTIONS]
            """
            
            report += f"""
            • Face Ratio (Height/Width): {avg_features['face_ratio']:.2f} (Ideal ≈ 1.62)
            • Facial Symmetry: {avg_features['symmetry']:.2f}/1.0
            • Feature Alignment: {avg_features['feature_alignment']:.2f}/1.0
            """
            
            report += """
            [ASSESSMENT]
            """
            
            report += self.generate_detailed_assessment(avg_beauty_score, avg_features)
        
        self.report_text.insert(tk.END, report)
    
    def generate_assessment(self, emotion_results, avg_beauty_score, num_faces):
        assessment = ""
        
        if num_faces == 0:
            assessment = "No faces detected in the image. Please try with a clearer image."
        else:
            # Emotion assessment
            emotions = [r['emotion'] for r in emotion_results]
            primary_emotion = max(set(emotions), key=emotions.count)
            
            if primary_emotion in ['Happy', 'Surprise']:
                assessment += "The subjects appear to be in a positive emotional state. "
            elif primary_emotion in ['Angry', 'Disgust', 'Fear']:
                assessment += "The subjects appear to be in a negative emotional state. "
            elif primary_emotion == 'Neutral':
                assessment += "The subjects appear to be in a neutral emotional state. "
            else:
                assessment += "The subjects show varied emotional states. "
            
            # Beauty assessment if available
            if avg_beauty_score > 0:
                if avg_beauty_score > 0.8:
                    assessment += "Excellent facial features with well-balanced proportions and symmetry."
                elif avg_beauty_score > 0.6:
                    assessment += "Good facial structure with some excellent features. Minor improvements possible."
                elif avg_beauty_score > 0.4:
                    assessment += "Average features with room for improvement in some areas."
                else:
                    assessment += "Some facial features could benefit from professional consultation."
        
        return assessment
    
    def generate_detailed_assessment(self, score, features):
        assessment = ""
        
        if score > 0.8:
            assessment += "• Exceptional facial harmony and proportions detected\n"
            assessment += "• Features are well-balanced and symmetrical\n"
            assessment += "• Skin appears healthy and well-maintained\n"
        elif score > 0.6:
            assessment += "• Good facial structure with some outstanding features\n"
            assessment += "• Minor proportional adjustments could enhance natural beauty\n"
            assessment += "• Consider professional skincare advice for optimal results\n"
        elif score > 0.4:
            assessment += "• Facial features show potential with room for improvement\n"
            assessment += "• Targeted treatments could help balance proportions\n"
            assessment += "• A dermatologist could help improve skin quality\n"
        else:
            assessment += "• Significant opportunities for improvement identified\n"
            assessment += "• Professional consultation recommended for personalized advice\n"
            assessment += "• Multiple approaches available to enhance natural features\n"
        
        # Add specific recommendations based on metrics
        assessment += "\n[SPECIFIC RECOMMENDATIONS]\n"
        
        if features['face_ratio'] < 1.5 or features['face_ratio'] > 1.7:
            assessment += "• Face shape could benefit from proportional balancing techniques\n"
        
        if features['symmetry'] < 0.7:
            assessment += "• Facial exercises may help improve symmetry\n"
        
        if features['skin_evenness'] < 0.6:
            assessment += "• Professional skincare treatments could improve skin texture\n"
        
        return assessment
    
    def get_skin_roi(self, frame, landmarks):
        """Get optimal skin region avoiding shadows and makeup"""
        x1, y1 = int(landmarks[10][0]), int(landmarks[10][1])
        x2, y2 = int(landmarks[151][0]), int(landmarks[151][1])
        
        roi_height = int(abs(y2 - y1) * 0.2)
        roi = frame[max(0,y1):min(frame.shape[0],y1+roi_height), 
                   max(0,x1):min(frame.shape[1],x2)]
        
        return roi if roi.size > 0 else None
    
    def analyze_skin_tone(self, roi):
        """Precise skin tone analysis using LAB color space"""
        if roi is None: 
            return "Unknown", "#FFFFFF"
        
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        avg_l = np.mean(l)
        
        # Fitzpatrick Scale classification with hex colors
        if avg_l > 160: 
            return "Very Light", "#FFE5CC"
        elif avg_l > 140: 
            return "Light", "#E0BB97"
        elif avg_l > 120: 
            return "Medium", "#BC8E6C"
        elif avg_l > 100: 
            return "Tan", "#9B6C53"
        elif avg_l > 80: 
            return "Dark", "#62462B"
        else: 
            return "Very Dark", "#3C281E"
    
    def calculate_distance(self, p1, p2):
        return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    
    def calculate_facial_symmetry(self, landmarks):
        left_points = [landmarks[i] for i in [33, 133, 362, 263]]
        right_points = [landmarks[i] for i in [263, 362, 33, 133]]
        
        total_diff = 0
        for lp, rp in zip(left_points, right_points):
            total_diff += self.calculate_distance(lp, rp)
        
        return 1 / (1 + total_diff/len(left_points))
    
    def extract_beauty_features(self, landmarks, skin_roi):
        face_width = self.calculate_distance(landmarks[454], landmarks[234])
        face_height = self.calculate_distance(landmarks[152], landmarks[10])
        face_ratio = face_height / face_width
        
        symmetry = self.calculate_facial_symmetry(landmarks)
        
        skin_evenness = 0
        if skin_roi is not None:
            gray = cv2.cvtColor(skin_roi, cv2.COLOR_BGR2GRAY)
            skin_evenness = 1 - (cv2.Laplacian(gray, cv2.CV_64F).var() / 1000)
        
        eye_center = ((landmarks[33][0] + landmarks[263][0])/2, 
                      (landmarks[33][1] + landmarks[263][1])/2)
        nose_tip = landmarks[4]
        mouth_center = ((landmarks[13][0] + landmarks[14][0])/2,
                        (landmarks[13][1] + landmarks[14][1])/2)
        alignment = 1 / (1 + self.calculate_distance(eye_center, nose_tip)) +  1 / (1 + self.calculate_distance(nose_tip, mouth_center))
        
        return {
            'face_ratio': round(face_ratio, 3),
            'symmetry': round(symmetry, 3),
            'skin_evenness': round(skin_evenness, 3),
            'feature_alignment': round(alignment, 3)
        }
    
    def predict_beauty_score(self, features):
        score = 0
        for key, weight in self.BEAUTY_WEIGHTS.items():
            score += features[key] * weight
        return round(min(max(score, 0), 1), 3)
    
    def draw_score_meter(self, score):
        # Draw background
        self.beauty_score_canvas.create_rectangle(0, 0, 300, 80, fill="white", outline="")
        
        # Draw meter
        self.beauty_score_canvas.create_rectangle(
            50, 30, 250, 60,
            outline="#cccccc",
            fill="#f0f0f0",
            width=2
        )
        
        # Calculate filled width (score from 0 to 1)
        fill_width = 200 * score
        fill_color = "#4CAF50" if score > 0.7 else "#FFC107" if score > 0.5 else "#F44336"
        
        # Draw filled portion
        self.beauty_score_canvas.create_rectangle(
            50, 30, 50 + fill_width, 60,
            outline=fill_color,
            fill=fill_color,
            width=2
        )
        
        # Draw markers and labels
        for i in range(0, 6):
            x = 50 + (i * 40)
            self.beauty_score_canvas.create_line(x, 30, x, 20, fill="#999999")
            self.beauty_score_canvas.create_text(x, 10, text=f"{i*0.2:.1f}", font=("Arial", 8))
        
        # Add score text
        self.beauty_score_canvas.create_text(
            150, 45,
            text=f"{score:.2f}",
            font=("Arial", 14, "bold"),
            fill="#333333"
        )
    
    def display_processed_image(self, image_array):
        # Convert numpy array to PIL Image
        image = Image.fromarray(image_array)
        image = ImageOps.fit(image, (400, 300), method=Image.LANCZOS)
        
        # Update canvas
        self.processed_photo = ImageTk.PhotoImage(image)
        self.processed_canvas.create_image(0, 0, anchor="nw", image=self.processed_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAnalysisApp(root)
    root.mainloop()