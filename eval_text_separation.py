import os
import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from scipy.linalg import sqrtm


def calculate_ssim(image1, image2):
    # Convert to grayscale and compute SSIM
    image1_array = np.array(image1.convert("L"))  
    image2_array = np.array(image2.convert("L"))
        
    ssim_value = ssim(image1_array, image2_array)
    return ssim_value

def calculate_psnr(image1, image2):
    # Convert images to RGB if they are grayscale
    if image1.mode != 'RGB':
        image1 = image1.convert("RGB")
    if image2.mode != 'RGB':
        image2 = image2.convert("RGB")
        
    image1_array = np.array(image1)
    image2_array = np.array(image2)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(image1_array.flatten(), image2_array.flatten())
    if mse == 0:
        return 100  # perfect match, no error
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value


def calculate_fid(pred_images, target_images, device):
    inception_model = models.inception_v3(pretrained=True).to(device)
    inception_model.eval()

    # Transform to tensor for Inception V3
    preprocess = transforms.Compose([
        transforms.Resize(299), # Inception V3 uses size 299
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    def get_inception_features(images):
        features = []
        with torch.no_grad():
            for img in images:
                img_tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
                feature = inception_model(img_tensor)
                features.append(feature.squeeze().cpu().numpy())
        return np.array(features)

    pred_features = get_inception_features([img[1] for img in pred_images])
    target_features = get_inception_features([img[1] for img in target_images])

    # Calculate the mean and covariance of the feature vectors
    pred_mean = np.mean(pred_features, axis=0)
    target_mean = np.mean(target_features, axis=0)
    pred_cov = np.cov(pred_features, rowvar=False)
    target_cov = np.cov(target_features, rowvar=False)

    # Calculate FID
    mean_diff = np.linalg.norm(pred_mean - target_mean)
    cov_sqrt = sqrtm(np.dot(pred_cov, target_cov))
    fid = mean_diff + np.trace(pred_cov + target_cov - 2 * cov_sqrt)
    return fid

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            with Image.open(img_path) as img:
                images.append((img_path, img.copy()))
    return images

def main():
    prediction_path = "DDPM/Gen_MNIST"
    target_path = "synthetic_MNIST/test/undertext"

    pred_images = load_images_from_folder(prediction_path)
    target_images = load_images_from_folder(target_path)

    pred_images.sort(key=lambda x: os.path.basename(x[0]))
    target_images.sort(key=lambda x: os.path.basename(x[0]))

    ssims = []
    psnrs = []

    for pred_img, target_img in zip(pred_images, target_images):
        ssim_value = calculate_ssim(pred_img[1], target_img[1])
        psnr_value = calculate_psnr(pred_img[1], target_img[1])
        ssims.append(ssim_value)
        psnrs.append(psnr_value)

    average_ssim = np.nanmean(ssims)
    average_psnr = np.nanmean(psnrs)

    print(f"Results for {prediction_path.split('/')[-2]}:")
    print(f"Average SSIM: {average_ssim}")
    print(f"Average PSNR: {average_psnr}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid_value = calculate_fid(pred_images, target_images, device)
    print(f"FID: {fid_value}")

if __name__ == "__main__":
    main()