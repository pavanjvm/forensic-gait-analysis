import numpy as np
import matplotlib.pyplot as plt

# Provided keypoint label
keypoint_label = "0 0.0611979 0.498843 0.122396 0.831019 0.0755574 0.152087 0.927237 0.078169 0.13905 0.689377 0.0666599 0.140598 0.913861 0 0 0.2195 0.0406246 0.154759 0.822373 0.0652732 0.248009 0.933115 0.0237151 0.256995 0.974628 0.0751536 0.374447 0.776236 0.00346748 0.402782 0.918363 0.0873975 0.48196 0.767014 0.00974127 0.538449 0.898391 0.055282 0.492017 0.929133 0.0349031 0.500406 0.949232 0.0424203 0.657628 0.878526 0.0651411 0.679172 0.911355 0.0191133 0.828399 0.660225 0.0666354 0.8558 0.695421"


# Parse the keypoint data
keypoint_data = np.array(keypoint_label.split(), dtype=float)
class_id = int(keypoint_data[0])  # Extract class ID (not used here)
confidence = keypoint_data[1]    # Extract confidence score (not used here)
keypoints = keypoint_data[2:].reshape(-1, 2)  # Extract (x, y) coordinates

# Path to the image frame (replace with the actual image path)
image_path = r'C:\Users\pavan\OneDrive\Desktop\keyppint.png'
image = plt.imread(image_path)

# Visualize keypoints on the image
plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.title(f"Keypoints Visualization (Class ID: {class_id}, Confidence: {confidence:.2f})")

# Plot keypoints
for idx, (x, y) in enumerate(keypoints):
    if x != 0 and y != 0:  # Skip keypoints with 0, indicating no detection
        plt.scatter(x * image.shape[1], y * image.shape[0], label=f"Keypoint {idx}")
        plt.text(x * image.shape[1], y * image.shape[0], f"{idx}", color="red", fontsize=8)

plt.legend()
plt.axis("off")
plt.show()
