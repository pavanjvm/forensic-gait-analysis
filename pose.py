from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n-pose.pt")  # Load pretrained pose model

# Specify the path to the video file
video1 = r'C:\Users\pavan\OneDrive\Desktop\gait anlaysis\forensic-gait-analysis\processed_videos\processed_video1.mp4'
video2 = r'C:\Users\pavan\OneDrive\Desktop\gait anlaysis\forensic-gait-analysis\processed_videos\processed_video2.mp4'
# Process the video to extract keypoints
#results = model(video1, save=True, save_txt=True)
results = model(video2, save= True ,save_txt=True)

# Keypoints will be saved in a .txt file for each frame in the output directory
print("Keypoint extraction completed.")
