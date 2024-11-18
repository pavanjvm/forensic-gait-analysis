# Forensic Gait Analysis  

Forensic Gait Analysis is a project that compares two video footages — one from a crime scene and another from a suspect — to identify if they belong to the same person based on their gait (walking pattern).

## Features  
- Preprocesses videos into 20 FPS and trims them to 6 seconds (120 frames total).  
- Compares two videos using pose estimation and gait feature analysis.  
- Outputs a similarity score and visualization for comparison.  

---

## Project Setup  

1. **Preprocessing**:  
   Ensure your videos are preprocessed into:  
   - 20 frames per second (FPS).  
   - 6 seconds duration (120 frames total).  

   Store the processed videos in a folder named `processed_videos`.  

2. **Install Dependencies**:  
   Install the required Python packages by running:  
   ```bash
   pip install -r requirements.txt
