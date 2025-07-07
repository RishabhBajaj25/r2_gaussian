import imageio.v2 as imageio
import os
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # Force it to use the Qt5 GUI backend

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

# Directory to save TIFFs
save_dir = "/media/rishabh/SSD_1/Data/UTokyo/processed_2_tif/"
os.makedirs(save_dir, exist_ok=True)

n_imgs = 470
file_template = "/media/rishabh/SSD_1/Data/UTokyo/raw/CR_20250618_170040_/{:05}.RAW"



for i in range(n_imgs):
    file_path = file_template.format(i)
    image = np.rot90(read_raw_xray(file_path, width=2352, height=2944, dtype=np.uint16), k=3)

    # # Save 16-bit TIFF
    # tiff_path_16bit = os.path.join(save_dir, f"CR_20250618_170040_{i:05}.tif")
    # imageio.imwrite(tiff_path_16bit, image, format="tiff")
    # print(f"Saved 16-bit TIFF: {tiff_path_16bit}")

    # Save normalized 8-bit TIFF (optional)
    image_norm = (image - image.min()) / (image.max() - image.min())
    image_8bit = (image_norm * 255).astype(np.uint16)
    tiff_path_8bit = os.path.join(save_dir, f"CR_20250618_170040_{i+1:04}.tif")
    imageio.imwrite(tiff_path_8bit, image_8bit, format="tiff")
    print(f"Saved 16-bit TIFF: {tiff_path_8bit}")

    # Display if you want
    # fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    # axes[0].imshow(image, cmap='gray')
    # axes[0].set_title('Original 16-bit X-ray')
    # axes[0].axis('off')
    # axes[1].imshow(image_8bit, cmap='gray')
    # axes[1].set_title('Normalized 8-bit X-ray')
    # axes[1].axis('off')
    # plt.tight_layout()
    # plt.show()
