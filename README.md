# Forensic Gait Analysis  

Forensic Gait Analysis is a project that compares two video footages — one from a crime scene and another from a suspect — to identify if they belong to the same person based on their gait (walking pattern).

## Features  
- Preprocesses videos into 20 FPS and trims them to 6 seconds (120 frames total).  
- Compares two videos using pose estimation and gait feature analysis.  
- Outputs a similarity score and visualization for comparison.  

---

## Project Setup  

### Step 1: Install Dependencies  

Install the required Python packages by running:  

```bash
pip install -r requirements.txt
```


### Step 2: Preprocessing  

In this step, you will preprocess your input video files. The preprocessing will:
- Convert videos to 20 frames per second (FPS).  
- Trim the videos to a 6-second duration (120 frames in total).

To begin, follow these steps:

1. **Place Your Videos**:  
   Place the video files you want to preprocess in a folder (for example, `raw_videos/`).

2. **Update Video Paths**:  
   Open the `preprocess.py` script and update the paths to point to your raw video files.  
   - Set the input video path to where your raw videos are located.  
   - Set the output folder to store the processed videos (e.g., `processed_videos/`).

3. **Run the Preprocessing Script**:  
   Run the following command to preprocess the videos:  

   ```bash
   python preprocess.py

