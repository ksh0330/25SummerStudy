import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Create sample images directory
images_dir = "cs231n/datasets/coco_captioning/images"
os.makedirs(images_dir, exist_ok=True)

# Generate diverse sample images
np.random.seed(231)

# Create different types of images
def create_sample_image(image_type, size=(224, 224)):
    """Create a sample image based on type"""
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    if image_type == "person":
        # Create a simple person silhouette
        img[50:200, 100:124] = [100, 150, 200]  # Body
        img[30:50, 110:114] = [150, 100, 100]   # Head
        img[50:120, 90:100] = [200, 150, 100]   # Left arm
        img[50:120, 124:134] = [200, 150, 100]  # Right arm
        img[200:224, 110:114] = [150, 100, 100] # Left leg
        img[200:224, 120:124] = [150, 100, 100] # Right leg
        
    elif image_type == "car":
        # Create a simple car
        img[100:140, 50:174] = [200, 100, 100]  # Body
        img[80:100, 70:154] = [150, 150, 200]   # Windshield
        img[140:160, 60:80] = [100, 100, 100]   # Wheel
        img[140:160, 144:164] = [100, 100, 100] # Wheel
        
    elif image_type == "dog":
        # Create a simple dog
        img[80:120, 100:140] = [150, 100, 50]   # Body
        img[60:80, 110:130] = [200, 150, 100]   # Head
        img[70:90, 90:100] = [200, 150, 100]    # Ear
        img[70:90, 140:150] = [200, 150, 100]  # Ear
        img[120:140, 110:130] = [150, 100, 50]  # Legs
        
    elif image_type == "bike":
        # Create a simple bike
        img[100:120, 100:124] = [100, 100, 100] # Frame
        img[80:100, 100:120] = [100, 100, 100]  # Handlebar
        img[120:140, 100:120] = [100, 100, 100] # Seat
        img[140:160, 100:120] = [100, 100, 100] # Pedal
        
    elif image_type == "tree":
        # Create a simple tree
        img[120:224, 110:114] = [100, 50, 0]    # Trunk
        img[80:120, 80:144] = [0, 150, 0]       # Leaves
        
    else:
        # Random colorful image
        img = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    
    return img

# Create sample images for train and val
image_types = ["person", "car", "dog", "bike", "tree"]

# Create train images (100)
for i in range(100):
    image_type = image_types[i % len(image_types)]
    img = create_sample_image(image_type)
    img_pil = Image.fromarray(img)
    img_pil.save(f"{images_dir}/train_{i:06d}.jpg")

# Create val images (20)
for i in range(20):
    image_type = image_types[i % len(image_types)]
    img = create_sample_image(image_type)
    img_pil = Image.fromarray(img)
    img_pil.save(f"{images_dir}/val_{i:06d}.jpg")

print(f"Created {100 + 20} sample images in {images_dir}")
print("Sample images created successfully!")
