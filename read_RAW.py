import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # Force it to use the Qt5 GUI backend

print(matplotlib.get_backend())
#  File path (use raw string for Windows path or escape backslashes)
file_path = "/media/rishabh/SSD_1/Data/XRay_Data/UTokyo/CR_20250618_170040/00000.RAW"

# Image dimensions
width = 2352
height = 2944

def read_raw_xray(filename, width, height, dtype=np.uint16, byteorder='little'):
    """Read raw X-ray image file"""

    # Calculate expected file size
    bytes_per_pixel = np.dtype(dtype).itemsize
    expected_size = width * height * bytes_per_pixel

    # Read the raw data
    with open(filename, 'rb') as f:
        raw_data = f.read()

    print(f"File size: {len(raw_data)} bytes")
    print(f"Expected size: {expected_size} bytes")

    # Convert to numpy array
    if byteorder == 'big':
        raw_data = np.frombuffer(raw_data, dtype=f'>{dtype.name}')
    else:
        raw_data = np.frombuffer(raw_data, dtype=dtype)

    # Reshape to image dimensions
    image = raw_data.reshape(height, width)

    return image

file_path =  "/media/rishabh/SSD_1/Data/UTokyo/CR_20250618_170040/00000.RAW"

n_imgs = 10
for i in range(n_imgs):
    file_path = file_path.replace(format(i,"05"),format(i+1,"05"))
    image = np.rot90(read_raw_xray(file_path, width=2352, height=2944, dtype=np.uint16), k=3)

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.colorbar()
    plt.title('X-ray Image')
    plt.show()


# # Enhanced visualization
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#
# # Original image
# axes[0].imshow(image, cmap='gray')
# axes[0].set_title('Original X-ray')
# axes[0].axis('off')
#
# # Contrast stretched to full range
# image_stretched = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
# axes[1].imshow(image_stretched, cmap='gray')
# axes[1].set_title('Contrast Stretched')
# axes[1].axis('off')
#
# # Histogram equalization for better contrast
# from skimage import exposure
# image_eq = exposure.equalize_hist(image)
# axes[2].imshow(image_eq, cmap='gray')
# axes[2].set_title('Histogram Equalized')
# axes[2].axis('off')
#
# plt.tight_layout()
# plt.show()
#
# # Show histogram
# plt.figure(figsize=(10, 4))
# plt.hist(image.flatten(), bins=100, alpha=0.7)
# plt.xlabel('Pixel Intensity')
# plt.ylabel('Frequency')
# plt.title('Intensity Histogram')
# plt.show()

# Print image statistics
print(f"Image shape: {image.shape}")
print(f"Data type: {image.dtype}")
print(f"Min value: {image.min()}")
print(f"Max value: {image.max()}")
print(f"Mean value: {image.mean():.2f}")