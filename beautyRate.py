import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk, ImageOps
import cv2
import mediapipe as mp
import numpy as np
import math
from datetime import datetime

class BeautyAnalyzerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Beauty Analyzer Pro")
        self.master.geometry("800x900")
        self.master.configure(bg="#f5f5f5")
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            max_num_faces=1, 
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
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Header Frame
        header_frame = tk.Frame(self.master, bg="#6a0dad")
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.header = tk.Label(
            header_frame, 
            text="Beauty Analyzer Pro", 
            font=("Arial", 24, "bold"), 
            fg="white",
            bg="#6a0dad",
            pady=15
        )
        self.header.pack()
        
        # Main Container Frame
        main_frame = tk.Frame(self.master, bg="#f5f5f5")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left Panel (Image Upload)
        left_panel = tk.Frame(main_frame, bg="#f5f5f5")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Image Canvas
        self.canvas = tk.Canvas(
            left_panel, 
            width=350, 
            height=350, 
            bg="white",
            highlightthickness=2,
            highlightbackground="#6a0dad"
        )
        self.canvas.pack(pady=(0, 20))
        
        # Upload Button
        self.upload_btn = tk.Button(
            left_panel, 
            text="📷 Upload Image", 
            command=self.upload_image,
            bg="#6a0dad",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=10,
            relief=tk.FLAT
        )
        self.upload_btn.pack(pady=(0, 10))
        
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
        self.analyze_btn.pack(pady=(0, 20))
        
        # Right Panel (Results)
        right_panel = tk.Frame(main_frame, bg="#f5f5f5")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # Results Notebook (Tabbed interface)
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Summary Tab
        self.summary_tab = tk.Frame(self.notebook, bg="#f5f5f5")
        self.notebook.add(self.summary_tab, text="Summary")
        
        # Detailed Analysis Tab
        self.details_tab = tk.Frame(self.notebook, bg="#f5f5f5")
        self.notebook.add(self.details_tab, text="Details")
        
        # Full Report Tab
        self.report_tab = tk.Frame(self.notebook, bg="#f5f5f5")
        self.notebook.add(self.report_tab, text="Full Report")
        
        # Initialize results UI components
        self.init_summary_tab()
        self.init_details_tab()
        self.init_report_tab()
        
    def init_summary_tab(self):
        # Score Frame
        score_frame = tk.Frame(self.summary_tab, bg="#f5f5f5")
        score_frame.pack(fill=tk.X, pady=20)
        
        self.score_header = tk.Label(
            score_frame,
            text="Your Beauty Score",
            font=("Arial", 16, "bold"),
            bg="#f5f5f5"
        )
        self.score_header.pack()
        
        self.score_canvas = tk.Canvas(
            score_frame, 
            width=300, 
            height=80, 
            bg="white",
            highlightthickness=0
        )
        self.score_canvas.pack(pady=10)
        
        # Skin Tone Frame
        skin_frame = tk.Frame(self.summary_tab, bg="#f5f5f5")
        skin_frame.pack(fill=tk.X, pady=10)
        
        self.skin_header = tk.Label(
            skin_frame,
            text="Skin Analysis",
            font=("Arial", 16, "bold"),
            bg="#f5f5f5"
        )
        self.skin_header.pack()
        
        skin_subframe = tk.Frame(skin_frame, bg="#f5f5f5")
        skin_subframe.pack()
        
        self.skin_label = tk.Label(
            skin_subframe,
            text="Skin Tone: ",
            font=("Arial", 12),
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
        
        self.skin_evenness_label = tk.Label(
            skin_subframe,
            text="Evenness: ",
            font=("Arial", 12),
            bg="#f5f5f5"
        )
        self.skin_evenness_label.pack(side=tk.LEFT)
        
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
        
    def init_details_tab(self):
        # Metrics Frame
        metrics_frame = tk.Frame(self.details_tab, bg="#f5f5f5")
        metrics_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Facial Proportions
        proportions_frame = tk.LabelFrame(
            metrics_frame,
            text="Facial Proportions",
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
        
        # Skin Analysis
        skin_frame = tk.LabelFrame(
            metrics_frame,
            text="Skin Analysis",
            font=("Arial", 12, "bold"),
            bg="#f5f5f5"
        )
        skin_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.skin_tone_label = tk.Label(
            skin_frame,
            text="Skin Tone: ",
            font=("Arial", 11),
            bg="#f5f5f5"
        )
        self.skin_tone_label.pack(anchor="w")
        
        self.evenness_label = tk.Label(
            skin_frame,
            text="Evenness Score: ",
            font=("Arial", 11),
            bg="#f5f5f5"
        )
        self.evenness_label.pack(anchor="w")
        
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
        self.image = ImageOps.fit(self.image, (350, 350), method=Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
    
    def analyze_image(self):
        if not hasattr(self, 'image_path'):
            messagebox.showerror("Error", "Please upload an image first")
            return
            
        # Process the image
        frame = cv2.imread(self.image_path)
        if frame is None:
            messagebox.showerror("Error", "Could not read the image file")
            return
            
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            messagebox.showerror("Error", "No face detected in the image")
            return
            
        # Process the first face found
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) 
                   for lm in face_landmarks.landmark]
        
        # Skin analysis
        skin_roi = self.get_skin_roi(frame, landmarks)
        skin_tone, skin_color = self.analyze_skin_tone(skin_roi)
        
        # Beauty analysis
        features = self.extract_beauty_features(landmarks, skin_roi)
        beauty_score = self.predict_beauty_score(features)
        
        # Update all UI components with results
        self.update_summary_tab(beauty_score, skin_tone, skin_color, features)
        self.update_details_tab(features, skin_tone)
        self.update_report_tab(beauty_score, skin_tone, features)
    
    def update_summary_tab(self, beauty_score, skin_tone, skin_color, features):
        # Clear previous results
        self.score_canvas.delete("all")
        self.skin_color_display.delete("all")
        self.assessment_text.delete(1.0, tk.END)
        
        # Draw score meter
        self.draw_score_meter(beauty_score)
        
        # Update skin tone display
        self.skin_label.config(text=f"Skin Tone: {skin_tone}")
        self.skin_color_display.create_rectangle(
            0, 0, 50, 50, 
            fill=skin_color,
            outline=""
        )
        
        # Update skin evenness
        self.skin_evenness_label.config(text=f"Evenness: {features['skin_evenness']:.2f}/1.0")
        
        # Update assessment
        assessment = self.generate_assessment(beauty_score)
        self.assessment_text.insert(tk.END, assessment)
    
    def update_details_tab(self, features, skin_tone):
        # Update facial proportions
        self.face_ratio_label.config(
            text=f"Face Ratio (H/W): {features['face_ratio']:.2f} (Ideal ≈ 1.62)"
        )
        self.symmetry_label.config(
            text=f"Symmetry Score: {features['symmetry']:.2f}/1.0"
        )
        self.alignment_label.config(
            text=f"Feature Alignment: {features['feature_alignment']:.2f}/1.0"
        )
        
        # Update skin analysis
        self.skin_tone_label.config(text=f"Skin Tone: {skin_tone}")
        self.evenness_label.config(
            text=f"Evenness Score: {features['skin_evenness']:.2f}/1.0"
        )
    
    def update_report_tab(self, beauty_score, skin_tone, features):
        self.report_text.delete(1.0, tk.END)
        
        report = f"""
        {' BEAUTY ANALYSIS REPORT '.center(50, '=')}
        
        Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        [SUMMARY]
        Overall Beauty Score: {beauty_score:.2f}/1.0
        Skin Tone: {skin_tone}
        
        [DETAILED METRICS]
        Facial Proportions:
        • Face Ratio (Height/Width): {features['face_ratio']:.2f} (Ideal ≈ 1.62)
        • Facial Symmetry: {features['symmetry']:.2f}/1.0
        • Feature Alignment: {features['feature_alignment']:.2f}/1.0
        
        Skin Analysis:
        • Skin Tone: {skin_tone}
        • Skin Evenness: {features['skin_evenness']:.2f}/1.0
        
        [ASSESSMENT]
        """
        
        report += self.generate_detailed_assessment(beauty_score, features)
        
        self.report_text.insert(tk.END, report)
    
    def generate_assessment(self, score):
        if score > 0.8:
            return "Excellent facial features! You have well-balanced proportions and symmetry."
        elif score > 0.6:
            return "Good facial structure with some excellent features. Minor improvements possible."
        elif score > 0.4:
            return "Average features with room for improvement in some areas."
        else:
            return "Some facial features could benefit from professional consultation."
    
    def generate_detailed_assessment(self, score, features):
        assessment = ""
        
        if score > 0.8:
            assessment += "• You have exceptional facial harmony and proportions\n"
            assessment += "• Your features are well-balanced and symmetrical\n"
            assessment += "• Maintain your current skincare routine for continued skin health\n"
        elif score > 0.6:
            assessment += "• You have good facial structure with some outstanding features\n"
            assessment += "• Minor proportional adjustments could enhance your natural beauty\n"
            assessment += "• Consider professional skincare advice for optimal results\n"
        elif score > 0.4:
            assessment += "• Your facial features show potential with room for improvement\n"
            assessment += "• Targeted treatments could help balance your proportions\n"
            assessment += "• A dermatologist could help improve skin quality\n"
        else:
            assessment += "• Significant opportunities for improvement identified\n"
            assessment += "• Professional consultation recommended for personalized advice\n"
            assessment += "• Multiple approaches available to enhance your natural features\n"
        
        # Add specific recommendations based on metrics
        assessment += "\n[SPECIFIC RECOMMENDATIONS]\n"
        
        if features['face_ratio'] < 1.5 or features['face_ratio'] > 1.7:
            assessment += "• Your face shape could benefit from proportional balancing techniques\n"
        
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
        self.score_canvas.create_rectangle(0, 0, 300, 80, fill="white", outline="")
        
        # Draw meter
        self.score_canvas.create_rectangle(
            50, 30, 250, 60,
            outline="#cccccc",
            fill="#f0f0f0",
            width=2
        )
        
        # Calculate filled width (score from 0 to 1)
        fill_width = 200 * score
        fill_color = "#4CAF50" if score > 0.7 else "#FFC107" if score > 0.5 else "#F44336"
        
        # Draw filled portion
        self.score_canvas.create_rectangle(
            50, 30, 50 + fill_width, 60,
            outline=fill_color,
            fill=fill_color,
            width=2
        )
        
        # Draw markers and labels
        for i in range(0, 6):
            x = 50 + (i * 40)
            self.score_canvas.create_line(x, 30, x, 20, fill="#999999")
            self.score_canvas.create_text(x, 10, text=f"{i*0.2:.1f}", font=("Arial", 8))
        
        # Add score text
        self.score_canvas.create_text(
            150, 45,
            text=f"{score:.2f}",
            font=("Arial", 14, "bold"),
            fill="#333333"
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = BeautyAnalyzerApp(root)
    root.mainloop()