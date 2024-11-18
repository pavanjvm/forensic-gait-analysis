from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n-pose.pt")  # Load pretrained pose model

# Specify the path to the video file
video1 = r'C:\Users\pavan\OneDrive\Desktop\gait anlaysis\forensic-gait-analysis\processed_videos\processed_video1.mp4'
video2 = r'C:\Users\pavan\OneDrive\Desktop\gait anlaysis\forensic-gait-analysis\processed_videos\processed_video2.mp4'
video3= r'C:\Users\pavan\OneDrive\Desktop\gait anlaysis\forensic-gait-analysis\processed_videos\processed_video3.mp4'
video4= r'C:\Users\pavan\OneDrive\Desktop\gait anlaysis\forensic-gait-analysis\processed_videos\processed_video4.mp4'

results = model(video1, save=True, save_txt=True)
results = model(video2, save= True ,save_txt=True)
results = model(video3, save = True, save_txt= True)
results = model(video4, save = True, save_txt= True)

print("Keypoint extraction completed.")
