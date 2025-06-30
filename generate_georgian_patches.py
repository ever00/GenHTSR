from PIL import Image
import numpy as np
import cv2
import os

def extract_patches(input_dir, output_dir, patch_size, undertext=False):
    '''
    Generate patches for the synthetic dataset simulating the real Georgian Palimpeset, created by [Jampour et al., 2024].
    '''
    os.makedirs(output_dir, exist_ok=True)
    
    patch_width, patch_height = patch_size

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            img_path = os.path.join(input_dir, filename)
            orig_img = Image.open(img_path)
            img = orig_img.copy()
            
            if undertext:
                # Extract the red undertext
                lower_red = np.array([205, 0, 0])
                upper_red = np.array([255, 100, 100])  
                red_mask = cv2.inRange(np.array(img), lower_red, upper_red)
                white_bg = np.ones_like(np.array(img), dtype=np.uint8) * 255
                red_result = np.where(red_mask[:, :, np.newaxis] == 255, np.array(img), white_bg)
                img = Image.fromarray(red_result)
            
            original_width, original_height = img.size
            print(f"Processing {filename}: Original Size = {original_width}x{original_height}")

            # Create patches of size 350x350
            patch_id = 0
            for y in range(0, original_height, 350):
                for x in range(0, original_width, 350):
                    box = (x, y, x + patch_width, y + patch_height)
                    patch = img.crop(box)

                    # Resize into 224x224
                    resized_patch = patch.resize((224,224))
                    patch_filename = f"{os.path.splitext(filename)[0]}_patch_{patch_id}.png"
                    resized_patch.save(os.path.join(output_dir, patch_filename))
                    patch_id += 1

            # Save cropped and resized patches
            print(f"Saved {patch_id} patches for {filename}")

if __name__ == '__main__':
    # Set paths
    input_dir = rf"...\SGP\train\combined"
    output_dir = rf"...\SGP\cropped\train\combined"

    # Extract patches
    extract_patches(input_dir, output_dir, (350,350), undertext=False)
