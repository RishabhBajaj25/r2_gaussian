import cv2
import numpy as np
import os

# Function to read the .npy file and visualize the images with a colormap
def visualize_images_from_npy(npy_file, step_size=4):
    # Load the .npy file containing the image stack
    image_data = np.load(npy_file)
    image_dir = npy_file.split('.')[0]
    os.makedirs(image_dir, exist_ok=True)

    # Loop through image data and for each step size, save the image using OpenCV
    for i in range(0, image_data.shape[0], step_size):
        # Extract the image
        image = image_data[i]

        # Convert the image to uint8 format if necessary
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Apply the 'plasma' colormap
        colored_image = cv2.applyColorMap(image, cv2.COLORMAP_PLASMA)

        # Rotate by 90 degrees clockwise
        colored_image = cv2.rotate(colored_image, cv2.ROTATE_90_CLOCKWISE)

        # Save the image
        image_name = os.path.join(image_dir, f'image_{i:03d}.png')
        cv2.imwrite(image_name, colored_image)
        print(f"Saved {image_name}")

def main():
    # Define the path to the .npy file
    npy_file = '/home/rishabh/projects/r2_gaussian/data/synthetic_dataset/cone_ntrain_75_angle_360/0_foot_cone/vol_gt.npy'

    # Call the function to visualize images from the .npy file
    visualize_images_from_npy(npy_file)

if __name__ == "__main__":
    main()
