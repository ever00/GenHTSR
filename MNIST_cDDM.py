import os
import sys
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image

torch.cuda.empty_cache()

sys.path.append("/proj/sciml/users/x_jesst")
from MNIST_data_loader import DataLoader


class ConditionalDiffusionUNet(nn.Module):
    def __init__(self, time_dim=128, T=30):
        '''
        U-Net setup
        '''
        super().__init__()
        self.time_embedding = nn.Embedding(T, time_dim)
        
        # Encoder
        self.conv1 = nn.Conv2d(3 + time_dim, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(512, 256, 3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, 3, padding=1)
        
        # Output
        self.out_conv = nn.Conv2d(64, 3, 3, padding=1)


    def forward(self, x, t):
        '''
        Forward pass through U-Net
        '''
        # Time embedding
        t_embed = self.time_embedding(t.long()).unsqueeze(-1).unsqueeze(-1)
        t_embed = t_embed.expand(-1, -1, x.shape[2], x.shape[3])
        
        # Input
        x = torch.cat([x, t_embed], dim=1)

        # Downsampel through the encoder
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))

        # Upsampel through the decoder
        x = F.relu(self.upconv1(x4))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.upconv3(x))

        # Output
        return torch.tanh(self.out_conv(x))

def get_cosine_schedule(T, device):
    '''
    Cosine noise scheduler
    '''
    # Steps for T
    steps = torch.linspace(0, T, T+1, device=device)

    # Cosine shedule
    f = torch.cos((steps / T + 0.008) / (1 + 0.008) * (math.pi / 2)) ** 2

    # Noise scaling over time
    betas = torch.clip(1 - (f[1:] / f[:-1]), 0.0001, 0.02) 
    return betas


def train_step(model, optimizer, combined_img, undertext_img, T, device):
    '''
    Training process
    '''
    model.train()

    # Get random time step
    t = torch.randint(0, T, (combined_img.size(0),), device=device).long()
    
    # Get noise scaling for T
    betas = get_cosine_schedule(T, device)
    beta_t = betas[t].view(-1, 1, 1, 1)
    alpha_t = torch.sqrt(1.0 - beta_t).view(-1, 1, 1, 1)

    # Noise calculated as everything except the undertext
    noise = combined_img - undertext_img

    # Scale the noise and add it to the clean undertext_img
    noisy_img = undertext_img + noise * alpha_t

    # Predict the noise
    pred_noise = model(noisy_img, t)

    # Calculate loss
    loss = F.l1_loss(pred_noise, noise)

    # Step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def denoise_image(model, combined_img, T, device):
    '''
    Denoise image by reversing noise process according to noise schedule
    '''
    model.eval() 
    
    # Get noise scaling for t
    betas = get_cosine_schedule(T, device)

    # Initial image is combined_img
    denoised_img = combined_img.clone()
    
    with torch.no_grad():

        # Start from T, max noise
        for t_step in reversed(range(T)): 
            
            # Create tensor for current T
            t = torch.full((combined_img.size(0),), t_step, device=device).float()

            # Predict current noise
            pred_noise = model(denoised_img, t)
            
            # Compute alpha_t for current beta
            beta_t = betas[t_step].view(-1, 1, 1, 1)
            alpha_t = torch.sqrt(1.0 - beta_t).view(-1, 1, 1, 1)

            # Denoise image
            if t_step > 0:
                denoised_img = (denoised_img - torch.sqrt(1.0 - alpha_t**2) * pred_noise) / alpha_t
            else:
                denoised_img = denoised_img - pred_noise

    # Return cleaned image when T=0
    return denoised_img


def denoise_image2(model, combined_img, T, device):
    '''
    Denoise image using one-step heuristic denoising approach
    '''
    model.eval() 

    # Create tensor for max T
    t = torch.full((combined_img.size(0),), 29, device=device).float()

    # Predict current noise
    pred_noise = model(combined_img, t)
 
    denoised_img = combined_img - pred_noise
    threshold = 0.9 # set threshold for heuristic denoising sensitivity
    mask = pred_noise.abs().mean(dim=1, keepdim=True) > threshold  # (B, 1, H, W)

    mask = mask.expand_as(denoised_img)

    # Set noisy regions to white (1.0), keep the rest unchanged
    denoised_img = torch.where(mask, torch.ones_like(denoised_img), denoised_img)
    return denoised_img


def plot_progress(combined_img, undertext_img, output_dir, epoch):
    '''
    Create and save plots for the models current performance (during training)
    '''
    model.eval() 

    output_dir = output_dir + "_progress"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad(): 
        fig, axes = plt.subplots(1, 3, figsize=(16, 16))

        # Slice for batch
        combined_img = combined_img[0:1]

        # Denoise combined_img
        denoised_img = denoise_image(model, combined_img, T, device)

        # Reverse normalization, detach, move to cpu, transpose and squeeze before saving
        combined_img = (0.5 * combined_img.detach().cpu().numpy() + 0.5)
        undertext_img = (0.5 * undertext_img.detach().cpu().numpy() + 0.5)
        output_img = (0.5 * denoised_img.detach().cpu().numpy() + 0.5)
        output_img = np.clip(output_img, 0, 1)

        axes[0].imshow(np.transpose(combined_img.squeeze(), (1, 2, 0)))  
        axes[0].set_title("Input")
        axes[1].imshow(np.transpose(undertext_img[0], (1, 2, 0)))  
        axes[1].set_title("Target")

        axes[2].imshow(np.array(Image.fromarray((np.transpose(output_img.squeeze(), (1, 2, 0)) * 255).astype(np.uint8)).convert("L")), cmap='gray')
        axes[2].set_title("Generated")
    
    # Save plot
    plt.savefig(f'{output_dir}/{epoch}.png')
    plt.close()  


def generate_and_save_test_set(output_dir):
    '''
    Generate cleaned up under-text images from combined test set
    '''
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # Load test data
    for idx, (combined_img, _, labels, indices) in enumerate(data_loader.load_batch(is_testing=True)):
        combined_img = torch.tensor(combined_img, dtype=torch.float32).permute(0, 3, 1, 2).to(device)

        with torch.no_grad():

            # Denoise images
            denoised_img = denoise_image(model, combined_img, T, device)
            output_img = (0.5 * denoised_img.detach().cpu().numpy() + 0.5)
            output_img = np.clip(output_img, 0, 1)
            denoised_img_output = (output_img * 255).astype(np.uint8)
            
            # Save images
            for i in range(len(combined_img)):
                img_name = os.path.join(output_dir, f"{indices[i]}_label{labels[i]}.png")
                img = denoised_img_output[i]
                img = img.transpose(1, 2, 0) 
                img_pil = Image.fromarray(img, mode="RGB")
                img_gray = img_pil.convert("L")
                img_gray.save(img_name)

        print(f"Processed batch {idx + 1}", flush=True)
    
    # Create labels
    create_gt_file(output_dir, "gt.txt")


def create_gt_file(output_dir, gt_filename):
    '''
    Generate .gt file with labels related to the generated under-text images (for AttehtionHTR)
    '''
    gt_path = os.path.join(output_dir, gt_filename)
    with open(gt_path, "w") as gt_file:
        for filename in os.listdir(output_dir):
            if filename.endswith(".png"):
                label = filename[-5:-4]
                gt_file.write(f"{filename} {label}\n")


T, num_epochs, lr = 30, 100, 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = 'MNIST_cDDM'

model = ConditionalDiffusionUNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

data_loader = DataLoader(dataset_name='synthetic_MNIST', img_res=(28, 28), batch_size=128)
print(f"DDPM, MNIST, epochs={num_epochs}, T={T}", flush=True)

for epoch in range(num_epochs):
    for idx, (combined_img, undertext_img, labels, original_idx) in enumerate(data_loader.load_batch()):
        
        combined_img = torch.tensor(combined_img, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        undertext_img = torch.tensor(undertext_img, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        loss = train_step(model, optimizer, combined_img, undertext_img, T, device)
        if idx % 10 == 0:
            print(f"Epoch {epoch+1} Batch {idx} Loss: {loss:.4f}", flush=True)

    plot_progress(combined_img, undertext_img, output_dir, epoch)
    scheduler.step()
    print(f"Learning Rate: {scheduler.get_last_lr()[0]}", flush=True)

# Generate clean under-text from combined test set    
generate_and_save_test_set(output_dir)
