import numpy as np
import matplotlib.pyplot as plt
import os

def load_keypoints(folder_path):
    """Load keypoint files from folder"""
    all_frames = []
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    files.sort()
    
    for file in files:
        with open(os.path.join(folder_path, file), 'r') as f:
            data = f.readline().strip().split()
            frame_data = [float(x) for x in data]
            all_frames.append(frame_data)
    
    return np.array(all_frames)

def extract_coordinates(keypoints):
    """Extract equal numbers of x and y coordinates"""
    # Ensure we have an even number of points
    if len(keypoints) % 2 != 0:
        keypoints = keypoints[:-1]  # Remove the last element if odd
    
    num_points = len(keypoints) // 2
    x_coords = keypoints[0:2*num_points:2]
    y_coords = keypoints[1:2*num_points:2]
    
    return x_coords, y_coords, num_points

def visualize_frame(frame_data, frame_num, save_path=None):
    """Visualize keypoints for a single frame"""
    keypoints = frame_data[7:]  # Skip first 7 values
    x_coords, y_coords, num_points = extract_coordinates(keypoints)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot keypoints
    plt.scatter(x_coords, y_coords, c='red', s=50, label='Keypoints')
    
    # Connect keypoints with lines to form skeleton
    # Only connect points that exist and are within our keypoint range
    for i in range(11, min(16, num_points - 1)):
        x = [x_coords[i], x_coords[i + 1]]
        y = [y_coords[i], y_coords[i + 1]]
        plt.plot(x, y, color='blue', linewidth=2, alpha=0.7)
    
    # Add labels for keypoints
    for i in range(11, min(17, num_points)):
        label = f'Point {i}'
        plt.annotate(label, 
                    (x_coords[i], y_coords[i]),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=8)
    
    plt.title(f'Frame {frame_num} Keypoint Verification')
    plt.grid(True)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_sequence(sequence_data, start_frame=0, num_frames=5):
    """Visualize multiple frames in sequence"""
    num_frames = min(num_frames, len(sequence_data) - start_frame)
    fig, axes = plt.subplots(1, num_frames, figsize=(20, 4))
    fig.suptitle('Keypoint Sequence Verification')
    
    if num_frames == 1:
        axes = [axes]
    
    for i in range(num_frames):
        frame_idx = start_frame + i
        frame = sequence_data[frame_idx]
        keypoints = frame[7:]
        
        x_coords, y_coords, num_points = extract_coordinates(keypoints)
        
        # Plot points
        axes[i].scatter(x_coords, y_coords, c='red', s=30)
        
        # Connect keypoints
        for j in range(11, min(16, num_points - 1)):
            x = [x_coords[j], x_coords[j + 1]]
            y = [y_coords[j], y_coords[j + 1]]
            axes[i].plot(x, y, color='blue', linewidth=1)
        
        axes[i].set_title(f'Frame {frame_idx}')
        axes[i].grid(True)
        axes[i].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def analyze_keypoint_data(sequence_data):
    """Analyze keypoint data for basic statistics and anomalies"""
    keypoints = sequence_data[0, 7:]  # Get first frame keypoints
    x_coords, y_coords, num_points = extract_coordinates(keypoints)
    
    print(f"\nData Analysis:")
    print(f"Total frames: {len(sequence_data)}")
    print(f"Keypoints per frame: {num_points}")
    print(f"Data shape: {sequence_data.shape}")
    
    # Analyze coordinate ranges across all frames
    all_x_coords = []
    all_y_coords = []
    
    for frame in sequence_data:
        keypoints = frame[7:]
        x, y, _ = extract_coordinates(keypoints)
        all_x_coords.extend(x)
        all_y_coords.extend(y)
    
    print("\nCoordinate Ranges:")
    print(f"X-coordinate range: [{min(all_x_coords):.2f}, {max(all_x_coords):.2f}]")
    print(f"Y-coordinate range: [{min(all_y_coords):.2f}, {max(all_y_coords):.2f}]")

def verify_keypoints(folder_path, output_folder=None):
    """Main verification function"""
    # Load data
    print(f"Loading keypoints from: {folder_path}")
    sequence_data = load_keypoints(folder_path)
    print(f"Loaded {len(sequence_data)} frames")
    
    # Analyze data
    analyze_keypoint_data(sequence_data)
    
    # Visualize single frames
    print("\nGenerating visualizations...")
    frames_to_check = [0, len(sequence_data)//2, -1]  # First, middle, and last frames
    
    for frame_idx in frames_to_check:
        if output_folder:
            save_path = os.path.join(output_folder, f'frame_{frame_idx}.png')
            visualize_frame(sequence_data[frame_idx], frame_idx, save_path)
        else:
            visualize_frame(sequence_data[frame_idx], frame_idx)
    
    # Visualize sequence
    print("\nGenerating sequence visualization...")
    visualize_sequence(sequence_data, start_frame=0, num_frames=5)
    
    return sequence_data

if __name__ == "__main__":
    folder_path = r"C:\Users\pavan\OneDrive\Desktop\gait anlaysis\forensic-gait-analysis\runs\pose\predict2\labels"
    output_folder = r"C:\Users\pavan\OneDrive\Desktop\gait anlaysis\forensic-gait-analysis\keypoints_verify\visualizations"
    
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    verify_keypoints(folder_path, output_folder)